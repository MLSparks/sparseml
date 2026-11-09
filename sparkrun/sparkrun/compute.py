"""Accelerated compute for the one heavy step agents hit: ranking large candidate sets.

Agents themselves are cheap. What gets expensive is scoring every combination, like
every flight against every hotel. best_pairs() does that on a GPU when CuPy or PyTorch
with CUDA is installed, otherwise with NumPy, otherwise in plain Python. Same answer
every way; nothing to configure.
"""
from __future__ import annotations

import math
from typing import Sequence

_BACKEND: str | None = None


def backend() -> str:
    """The fastest available backend: "cupy", "torch-cuda", "numpy" or "python"."""
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = "python"
        try:
            import cupy  # noqa: F401
            if cupy.cuda.runtime.getDeviceCount() > 0:
                _BACKEND = "cupy"
        except Exception:
            pass
        if _BACKEND == "python":
            try:
                import torch
                if torch.cuda.is_available():
                    _BACKEND = "torch-cuda"
            except Exception:
                pass
        if _BACKEND == "python":
            try:
                import numpy  # noqa: F401
                _BACKEND = "numpy"
            except Exception:
                pass
    return _BACKEND


def best_pairs(a_cost: Sequence[float], a_score: Sequence[float], b_cost: Sequence[float],
               b_score: Sequence[float], budget: float, *, b_mult: float = 1.0, k: int = 5,
               use: str | None = None) -> list[tuple[int, int, float, float]]:
    """For each item i in A, find the best j in B with a_cost[i] + b_mult*b_cost[j] <= budget,
    maximizing a_score[i] + b_score[j]. Return the top k as (i, j, score, total_cost).

    Ties go to the lowest index, on every backend.
    """
    how = use or backend()
    if how == "python" or not len(a_cost) or not len(b_cost):
        return _python(a_cost, a_score, b_cost, b_score, budget, b_mult, k)
    if how == "torch-cuda":
        return _torch(a_cost, a_score, b_cost, b_score, budget, b_mult, k)
    if how == "cupy":
        import cupy as xp
    else:
        import numpy as xp
    ac, as_ = xp.asarray(a_cost, dtype=xp.float64), xp.asarray(a_score, dtype=xp.float64)
    bc, bs = xp.asarray(b_cost, dtype=xp.float64) * b_mult, xp.asarray(b_score, dtype=xp.float64)
    rows = max(1, (1 << 22) // len(b_cost))  # bound memory: ~4M cells per chunk
    best_s, best_j = [], []
    for r in range(0, len(a_cost), rows):
        ok = (ac[r:r + rows, None] + bc[None, :]) <= budget
        s = xp.where(ok, as_[r:r + rows, None] + bs[None, :], -xp.inf)
        j = xp.argmax(s, axis=1)
        best_j.append(j)
        best_s.append(s[xp.arange(len(j)), j])
    best_s, best_j = xp.concatenate(best_s), xp.concatenate(best_j)
    order = xp.argsort(-best_s, kind="stable")[:k]
    out = []
    for i in (order.get() if how == "cupy" else order).tolist():
        sc = float(best_s[i])
        if math.isinf(sc):
            break
        j = int(best_j[i])
        out.append((i, j, sc, float(ac[i] + bc[j])))
    return out


def _torch(a_cost, a_score, b_cost, b_score, budget, b_mult, k):
    import torch
    dev = "cuda"
    ac = torch.tensor(a_cost, dtype=torch.float64, device=dev)
    as_ = torch.tensor(a_score, dtype=torch.float64, device=dev)
    bc = torch.tensor(b_cost, dtype=torch.float64, device=dev) * b_mult
    bs = torch.tensor(b_score, dtype=torch.float64, device=dev)
    s = torch.where(ac[:, None] + bc[None, :] <= budget, as_[:, None] + bs[None, :],
                    torch.tensor(-math.inf, dtype=torch.float64, device=dev))
    best_s, best_j = s.max(dim=1)  # first max on ties
    order = torch.sort(-best_s, stable=True).indices[:k]
    out = []
    for i in order.tolist():
        sc = float(best_s[i])
        if math.isinf(sc):
            break
        j = int(best_j[i])
        out.append((i, j, sc, float(ac[i] + bc[j])))
    return out


def _python(a_cost, a_score, b_cost, b_score, budget, b_mult, k):
    best = []
    for i, (ac, asc) in enumerate(zip(a_cost, a_score)):
        bj, bsc = -1, -math.inf
        for j, (bc, bsj) in enumerate(zip(b_cost, b_score)):
            if ac + bc * b_mult <= budget and asc + bsj > bsc:
                bj, bsc = j, asc + bsj
        if bj >= 0:
            best.append((i, bj, bsc, ac + b_cost[bj] * b_mult))
    best.sort(key=lambda t: -t[2])  # stable: lower i first on ties
    return best[:k]
