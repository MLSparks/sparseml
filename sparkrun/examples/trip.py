"""Multi-agent trip booking with sparkrun.

    python examples/trip.py                 # offline, deterministic stand-in models
    python examples/trip.py --approve       # approve the booking automatically
    python examples/trip.py --options 3000  # rank 3000 flights x 3000 hotels (uses a GPU if present)

Five specialist agents share a blackboard:

    flights ─┐
             ├─► rank (GPU if available) ─┬─► activities ─┐
    hotels ──┘                            └───────────────┴─► booking (needs approval)

Every agent runs under sparkrun: independent checks, crash-safe journals, side effects at
most once. Booking spends money, so a policy gate pauses it until a person approves.
At the end, team checks verify the whole trip: one flight and one hotel booked, the hotel
check-in matches the flight's arrival, activities fall inside the stay, and the total
fits the budget.

The stand-in models are small deterministic policies, so the demo is free and repeatable.
Swap in OpenAICompatible models to run the same team on real small models.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparkrun import Agent, Policy, Role, ScriptedModel, Team, compute, stable_hash, tool, verify

AIRLINES = ["Atlas Air", "Brightjet", "Coastline", "Delta Wing", "Eastwind", "Fjord Air"]
AREAS = ["Old Town", "Riverside", "Harbor", "Hillside", "Market Quarter", "University"]
# A small travel guide the Guide agent searches. Its answers are graded by ARES groundedness:
# they must be supported by the passages it actually retrieved.
GUIDES = {
    "Lisbon": [
        ("Money", "Portugal uses the euro. Cards are widely accepted, but small cafes may prefer cash."),
        ("Tram 28", "Tram 28 runs through the Alfama, Baixa and Graca districts, the classic way to see the old town."),
        ("Belem Tower", "Belem Tower is a 16th-century fortification on the Tagus and a UNESCO World Heritage Site."),
        ("Lisboa Card", "The Lisboa Card covers public transport and free or discounted entry to many museums."),
        ("Pasteis de Belem", "Pasteis de Belem has been baking custard tarts in Belem since 1837."),
        ("Fado", "Fado, Lisbon's traditional music, is on UNESCO's list of Intangible Cultural Heritage."),
    ],
}
GUIDE_QUESTIONS = {"Lisbon": ("What currency does Portugal use, and which tram line runs through Alfama?", "euro, Tram 28")}

PRICES = {"small": 0.0002, "strong": 0.003}  # stand-in models' cost per call, in dollars

ACTIVITIES = [  # (name, interest, price per person)
    ("Free walking tour of the old town", "history", 0), ("City history museum", "history", 18),
    ("Castle and viewpoints", "history", 15), ("Street food crawl", "food", 45),
    ("Market breakfast tasting", "food", 25), ("Cooking class", "food", 70),
    ("Sunset coastal hike", "nature", 0), ("Botanical garden", "nature", 8),
    ("Day trip to the national park", "nature", 55), ("Live music night", "music", 30),
    ("Contemporary art gallery", "art", 12), ("Street art bike tour", "art", 35),
]


@dataclass
class TripRequest:
    origin: str = "SFO"
    destination: str = "Lisbon"
    depart: str = "2026-12-18"
    nights: int = 5
    travelers: int = 2
    budget: float = 4200.0
    interests: list = field(default_factory=lambda: ["food", "history"])
    max_stops: int = 1
    options: int = 800


# --------------------------------------------------------------------------- catalog

def _rng(req: TripRequest, salt: str) -> random.Random:
    return random.Random(stable_hash([req.origin, req.destination, req.depart, req.options, salt]))


def search_flights_catalog(req: TripRequest) -> list[dict]:
    r = _rng(req, "flights")
    out = []
    for i in range(req.options):
        stops = r.choices([0, 1, 2], [0.35, 0.45, 0.2])[0]
        hours = round(r.uniform(9, 13) + stops * r.uniform(1.5, 4), 1)
        dep_h = r.randint(6, 23)
        price = round(r.uniform(420, 1350) - (dep_h > 21) * 60 + stops * -90, 0)
        arrive = dt.date.fromisoformat(req.depart) + dt.timedelta(days=int((dep_h + hours + 8) // 24))
        out.append({"id": f"F{i:04d}", "airline": r.choice(AIRLINES), "price": max(price, 280.0),
                    "stops": stops, "hours": hours, "depart_hour": dep_h, "arrive": arrive.isoformat()})
    return [f for f in out if f["stops"] <= req.max_stops]


def search_hotels_catalog(req: TripRequest) -> list[dict]:
    r = _rng(req, "hotels")
    return [{"id": f"H{i:04d}", "name": f"{r.choice(['Casa', 'Hotel', 'Pousada', 'Suites'])} {r.choice(AREAS)} {i}",
             "area": r.choice(AREAS), "price": round(r.uniform(65, 420), 0),
             "rating": round(r.uniform(3.0, 5.0), 1), "km": round(r.uniform(0.2, 9.0), 1)}
            for i in range(req.options)]


def flight_score(f: dict) -> float:
    return 100 - f["price"] / 25 - f["stops"] * 12 - f["hours"] * 1.2


def hotel_score(h: dict) -> float:
    return h["rating"] * 22 - h["km"] * 2.5 - h["price"] / 40


# --------------------------------------------------------------------------- stand-in models

def stand_in(name: str, decide, usd_per_call: float = 0.0002) -> ScriptedModel:
    """A deterministic policy that reads the prompt like a model would and returns one action."""
    return ScriptedModel(lambda messages: json.dumps(decide(messages[1]["content"])),
                         name=name, usd_per_call=usd_per_call)


def _request(prompt: str) -> dict:
    m = re.search(r"REQUEST: (\{.*?\})\s*$", prompt, re.M)
    return json.loads(m.group(1)) if m else {}


def _did(prompt: str, tool_name: str) -> bool:
    return f"{tool_name}(" in prompt and "-> OK" in prompt.split(f"{tool_name}(", 1)[1].split("\n", 1)[0]


# --------------------------------------------------------------------------- team

def build_team(req: TripRequest, runs_dir: str | Path, *, board: dict | None = None,
               gate=None, models: dict | None = None, on_event=None, guide: bool = False,
               wrap=None) -> Team:
    """models: optional {role: [model, ...]} to use real models instead of stand-ins.
    guide: add a Guide agent that answers a traveler question from a travel guide (graded by ARES).
    wrap: optional function applied to every model, e.g. lambda m: CachedModel(m, shared_cache)."""
    board = board if board is not None else {}
    board.setdefault("request", asdict(req))
    req_json = json.dumps(asdict(req))
    p = dict(backoff_s=0.2)

    @tool
    def search_flights(origin: str, destination: str, date: str, max_stops: int):
        """Search flights."""
        flights = search_flights_catalog(TripRequest(**{**asdict(req), "origin": origin, "destination": destination,
                                                        "depart": date, "max_stops": int(max_stops)}))
        board["flights"] = flights
        cheapest = min((f["price"] for f in flights), default=0)
        return f"{len(flights)} flights with at most {max_stops} stops; cheapest ${cheapest:.0f} per person"

    @tool
    def search_hotels(city: str, check_in: str, nights: int):
        """Search hotels."""
        hotels = search_hotels_catalog(req)
        board["hotels"] = hotels
        return f"{len(hotels)} hotels in {city} for {nights} nights from {check_in}"

    @tool
    def rank_itineraries(budget: float, top: int = 5):
        """Score every flight x hotel pair and keep the best options that fit the budget."""
        flights, hotels = board["flights"], board["hotels"]
        t0 = time.perf_counter()
        pre = board.pop("accelerated_rank", None)  # a browser GPU can do this step instead
        if pre:
            pairs, how = [tuple(x) for x in pre["pairs"]], pre["backend"]
        else:
            pairs = compute.best_pairs([f["price"] * req.travelers for f in flights], [flight_score(f) for f in flights],
                                       [h["price"] for h in hotels], [hotel_score(h) for h in hotels],
                                       float(budget) * 0.9, b_mult=req.nights, k=int(top))
            how = compute.backend()
        ms = pre["ms"] if pre else (time.perf_counter() - t0) * 1000
        board["options"] = [{"flight": flights[i], "hotel": hotels[j], "score": round(s, 2), "cost": round(c, 2)}
                            for i, j, s, c in pairs]
        board["rank_stats"] = {"backend": how, "ms": round(ms, 2), "pairs": len(flights) * len(hotels)}
        if not pairs:
            return "no flight and hotel combination fits the budget"
        return (f"ranked {len(flights) * len(hotels):,} combinations on {how} in {ms:.1f} ms; "
                f"best total ${pairs[0][3]:.0f}")

    @tool
    def plan_activities(interests: list, budget_left: float):
        """Plan one activity per full day within the remaining budget, free ones first."""
        opt = board["options"][0]
        start = dt.date.fromisoformat(opt["flight"]["arrive"])
        picks = sorted([a for a in ACTIVITIES if a[1] in interests], key=lambda a: a[2])
        plan, spent = [], 0.0
        for day in range(1, req.nights):
            for name, interest, price in picks:
                cost = price * req.travelers
                if name not in [p_["name"] for p_ in plan] and spent + cost <= budget_left:
                    plan.append({"day": (start + dt.timedelta(days=day)).isoformat(), "name": name,
                                 "interest": interest, "cost": cost})
                    spent += cost
                    break
        board["activities"] = plan
        return f"{len(plan)} activities for ${spent:.0f}"

    @tool(side_effect=True)
    def book_flight(flight_id: str):
        """Book a flight for all travelers."""
        f = next(f for f in board["flights"] if f["id"] == flight_id)
        board.setdefault("bookings", []).append({"type": "flight", "id": flight_id, "cost": f["price"] * req.travelers,
                                                 "arrive": f["arrive"], "ref": "FL-" + stable_hash(flight_id)[:6].upper()})
        return f"booked {flight_id} ({f['airline']}), arriving {f['arrive']}"

    @tool(side_effect=True)
    def book_hotel(hotel_id: str, check_in: str, nights: int):
        """Book a hotel."""
        h = next(h for h in board["hotels"] if h["id"] == hotel_id)
        board.setdefault("bookings", []).append({"type": "hotel", "id": hotel_id, "cost": h["price"] * int(nights),
                                                 "check_in": check_in, "nights": int(nights),
                                                 "ref": "HT-" + stable_hash(hotel_id)[:6].upper()})
        return f"booked {h['name']} from {check_in} for {nights} nights"

    # stand-in policies -----------------------------------------------------
    def flights_policy(prompt):
        r = _request(prompt)
        if _did(prompt, "search_flights"):
            return {"done": "Found flights"}
        return {"tool": "search_flights", "args": {"origin": r["origin"], "destination": r["destination"],
                                                   "date": r["depart"], "max_stops": r["max_stops"]}}

    def hotels_policy(prompt):
        r = _request(prompt)
        if _did(prompt, "search_hotels"):
            return {"done": "Found hotels"}
        return {"tool": "search_hotels", "args": {"city": r["destination"], "check_in": r["depart"], "nights": r["nights"]}}

    def rank_policy(prompt):
        if _did(prompt, "rank_itineraries"):
            return {"done": "Ranked options"}
        return {"tool": "rank_itineraries", "args": {"budget": _request(prompt)["budget"], "top": 5}}

    def lazy_activities(prompt):  # a cheap model that claims success without doing the work
        return {"done": "Planned great activities"}

    def activities_policy(prompt):
        r = _request(prompt)
        if _did(prompt, "plan_activities"):
            return {"done": f"Planned {len(board.get('activities', []))} activities inside the stay, within budget"}
        left = r["budget"] - board["options"][0]["cost"]
        return {"tool": "plan_activities", "args": {"interests": r["interests"], "budget_left": round(left, 2)}}

    def booking_policy(prompt):
        opt = board["options"][0]
        if not _did(prompt, "book_flight"):
            return {"tool": "book_flight", "args": {"flight_id": opt["flight"]["id"]}}
        if not _did(prompt, "book_hotel"):
            return {"tool": "book_hotel", "args": {"hotel_id": opt["hotel"]["id"], "check_in": opt["flight"]["arrive"],
                                                   "nights": req.nights}}
        return {"done": "Booked flight and hotel"}

    models = models or {}
    small, strong = PRICES["small"], PRICES["strong"]

    def agent(role, tools, policy, checks, fallback=None, gated=False):
        ms = models.get(role) or ([stand_in(f"{role}-small", policy, small)] if fallback is None else
                                  [stand_in(f"{role}-small", fallback, small), stand_in(f"{role}-strong", policy, strong)])
        ms = [wrap(m) for m in ms] if wrap else ms
        return Agent(ms, tools, checks=checks, runs_dir=runs_dir,
                     policy=Policy(max_steps=10, gate=gate if gated else None, **p))

    has = lambda key, n=1: verify.predicate(lambda c: (len(board.get(key, [])) >= n, f"{len(board.get(key, []))} {key}"),
                                            f"{key} found")
    roles = [
        Role("flights", agent("flights", [search_flights], flights_policy, [has("flights")]),
             f"Find flights for this trip.\nREQUEST: {req_json}"),
        Role("hotels", agent("hotels", [search_hotels], hotels_policy, [has("hotels")]),
             f"Find hotels for this trip.\nREQUEST: {req_json}"),
        Role("rank", agent("rank", [rank_itineraries], rank_policy, [has("options")]),
             f"Rank flight and hotel combinations within budget.\nREQUEST: {req_json}", after=("flights", "hotels")),
        Role("activities", agent("activities", [plan_activities], activities_policy,
                                 [has("activities", min(2, req.nights - 1))], fallback=lazy_activities),
             f"Plan activities for the stay.\nREQUEST: {req_json}", after=("rank",)),
        Role("booking", agent("booking", [book_flight, book_hotel], booking_policy,
                              [verify.predicate(lambda c: (len(board.get("bookings", [])) == 2,
                                                           f"{len(board.get('bookings', []))} bookings"), "flight and hotel booked")],
                              gated=True),
             lambda b: f"Book the best option.\nREQUEST: {req_json}\nOPTION: {json.dumps(b['options'][0])[:400]}",
             after=("rank", "activities")),
    ]

    if guide and req.destination in GUIDES:
        question, gold = GUIDE_QUESTIONS[req.destination]
        passages = GUIDES[req.destination]

        @tool
        def search_guide(query: str, k: int = 2):
            """Search the travel guide for the destination; returns the best passages."""
            words = set(re.findall(r"[a-z0-9]+", query.lower()))
            hits = sorted(passages, key=lambda p_: -len(words & set(re.findall(r"[a-z0-9]+", " ".join(p_).lower()))))[:int(k)]
            board.setdefault("guide_evidence", []).extend(text for _, text in hits)
            return [{"title": t, "text": x} for t, x in hits]

        def from_memory(prompt):  # a cheap model answering without looking anything up
            return {"done": "Portugal uses the escudo, and Tram 15 goes through Alfama."}

        def with_guide(prompt):
            if not _did(prompt, "search_guide"):
                return {"tool": "search_guide", "args": {"query": "Portugal currency euro tram Alfama", "k": 2}}
            return {"done": "Portugal uses the euro, and Tram 28 runs through Alfama."}

        def answer_saved(c):
            board["guide_answer"] = c["answer"]
            return True, c["answer"]

        roles.append(Role("guide", agent("guide", [search_guide], with_guide,
                                         [verify.grounded(lambda c: " ".join(board.get("guide_evidence", [])), min_score=0.8),
                                          verify.answer_f1(gold, min_score=0.4),
                                          verify.predicate(answer_saved, "answer recorded")],
                                         fallback=from_memory),
                          f"Answer the traveler's question using the travel guide.\nQUESTION: {question}"))

    def trip_checks():
        def one_each(c):
            types = sorted(b["type"] for b in board.get("bookings", []))
            return types == ["flight", "hotel"], f"bookings: {types}"

        def dates_line_up(c):
            bk = {b["type"]: b for b in board.get("bookings", [])}
            ok = "flight" in bk and "hotel" in bk and bk["hotel"]["check_in"] == bk["flight"]["arrive"]
            return ok, f"arrive {bk.get('flight', {}).get('arrive')}, check in {bk.get('hotel', {}).get('check_in')}"

        def in_stay(c):
            bk = {b["type"]: b for b in board.get("bookings", [])}
            if "hotel" not in bk:
                return False, "no hotel"
            start = dt.date.fromisoformat(bk["hotel"]["check_in"])
            end = start + dt.timedelta(days=bk["hotel"]["nights"])
            days = [dt.date.fromisoformat(a["day"]) for a in board.get("activities", [])]
            return all(start <= d < end for d in days), f"{len(days)} activities between {start} and {end}"

        def fits_budget(c):
            total = sum(b["cost"] for b in board.get("bookings", [])) + sum(a["cost"] for a in board.get("activities", []))
            board["total"] = round(total, 2)
            return total <= req.budget, f"${total:,.0f} of ${req.budget:,.0f}"

        return [verify.predicate(one_each, "exactly one flight and one hotel"),
                verify.predicate(dates_line_up, "hotel check-in matches flight arrival"),
                verify.predicate(in_stay, "activities inside the stay"),
                verify.predicate(fits_budget, "total within budget")]

    return Team(roles, board=board, checks=trip_checks(), runs_dir=runs_dir, on_event=on_event)


def spending_gate(tool_name: str, args: dict) -> str | None:
    """Stand-in for a company policy: anything that books (spends money) needs a person."""
    return f"{tool_name} spends money; a person must approve it" if tool_name.startswith("book_") else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--approve", action="store_true", help="approve the booking without asking")
    ap.add_argument("--options", type=int, default=800, help="flights and hotels to search (each)")
    args = ap.parse_args()
    req = TripRequest(options=args.options)
    runs = Path(tempfile.mkdtemp(prefix="sparkrun-trip-"))

    def show(role, e):
        if e["kind"] in ("tool_result", "escalated", "verified", "approval_requested", "finished"):
            text = {"tool_result": lambda: e["result"], "escalated": lambda: f"escalated to {e['model']}: {e['reason']}",
                    "verified": lambda: ("checks passed" if e["passed"] else "checks FAILED: " +
                                         "; ".join(ev["detail"] for ev in e["evidence"] if not ev["passed"])),
                    "approval_requested": lambda: f"waiting for approval: {e.get('note')}",
                    "finished": lambda: e["status"]}[e["kind"]]()
            print(f"  [{role:<10}] {text}")

    team = build_team(req, runs, gate=spending_gate, on_event=show)
    print(f"Planning {req.nights} nights in {req.destination} for {req.travelers}, budget ${req.budget:,.0f}\n")
    result = team.run("trip")
    if result.status == "needs_approval":
        opt = team.board["options"][0]
        print(f"\nBest option: {opt['flight']['airline']} {opt['flight']['id']} + {opt['hotel']['name']}, "
              f"${opt['cost']:,.0f} for flights and hotel")
        if args.approve or input("Approve booking? [y/N] ").strip().lower() == "y":
            for _ in range(2):  # flight, then hotel
                team.approve("trip", "booking")
                result = team.run("trip")
                if result.status != "needs_approval":
                    break
    print(f"\nTeam status: {result.status}   steps: {result.steps}   model cost: ${result.cost_usd:.4f}")
    if result.verdict:
        for ev in result.verdict.evidence:
            print(f"  {'PASS' if ev.passed else 'FAIL'}  {ev.check}: {ev.detail}")
    stats = team.board.get("rank_stats", {})
    print(f"Ranking: {stats.get('pairs', 0):,} combinations on {stats.get('backend')} in {stats.get('ms')} ms")


if __name__ == "__main__":
    main()
