"""Tools an agent can call, and the policy that limits what it may do."""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable


class TransientError(Exception):
    """A failure that may succeed on retry: timeouts, rate limits, 5xx responses."""


@dataclass
class Tool:
    name: str
    fn: Callable[..., Any]
    description: str = ""
    side_effect: bool = False
    """True if calling it changes the outside world (sends, writes, pays). Never run twice."""
    needs_approval: bool = False
    """True if a person must approve each distinct call before it runs."""
    params: list[str] = field(default_factory=list)

    def spec(self) -> str:
        flags = []
        if self.side_effect:
            flags.append("changes things; runs once")
        if self.needs_approval:
            flags.append("needs human approval")
        extra = f" [{'; '.join(flags)}]" if flags else ""
        return f"- {self.name}({', '.join(self.params)}): {self.description}{extra}"

    def check_args(self, args: dict) -> str | None:
        try:
            inspect.signature(self.fn).bind(**args)
            return None
        except TypeError as e:
            return f"Wrong arguments for {self.name}: {e}. Expected: {', '.join(self.params) or 'no arguments'}."

    def __call__(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)


def tool(fn: Callable | None = None, *, name: str | None = None, description: str | None = None,
         side_effect: bool = False, needs_approval: bool = False):
    """Turn a function into a Tool. Use bare (@tool) or with options (@tool(side_effect=True))."""

    def wrap(f: Callable) -> Tool:
        params = list(inspect.signature(f).parameters)
        desc = description if description is not None else (inspect.getdoc(f) or "").split("\n")[0]
        return Tool(name or f.__name__, f, desc, side_effect, needs_approval, params)

    return wrap(fn) if fn is not None else wrap


@dataclass
class Policy:
    """Hard limits for a run. The model cannot change these."""

    max_steps: int = 30
    max_cost_usd: float | None = None
    max_seconds: float | None = None
    max_tool_retries: int = 2
    max_model_retries: int = 3
    max_verify_failures: int = 3
    repeat_limit: int = 3
    """The same call this many times in a short window counts as stuck."""
    allowed_tools: set[str] | None = None
    backoff_s: float = 0.5
    gate: Callable[[str, dict], "str | None"] | None = None
    """Outside policy hook: gate(tool_name, args) returns None to allow, or a reason to
    pause the run for human approval (for example a company's spending rules)."""
