"""Model adapters. Lean by default: any OpenAI-compatible server, including local small models."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence, Union

from .tools import TransientError


class PermanentError(Exception):
    """A model failure that retrying will not fix (bad request, auth, missing model)."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Model(Protocol):
    name: str

    def complete(self, messages: list[dict], max_tokens: int = 512) -> tuple[str, Usage]: ...


class OpenAICompatible:
    """Any server that speaks the OpenAI chat API.

    Local and free: Ollama (http://localhost:11434/v1), llama.cpp server, vLLM, LM Studio.
    Hosted: pass api_key_env with the name of the environment variable that holds the key,
    and the prices so cost per verified task is tracked.
    """

    def __init__(self, base_url: str, model: str, *, api_key_env: str | None = None,
                 usd_per_mtok_in: float = 0.0, usd_per_mtok_out: float = 0.0,
                 temperature: float = 0.0, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.name = model
        self.api_key_env = api_key_env
        self.usd_in = usd_per_mtok_in
        self.usd_out = usd_per_mtok_out
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, messages: list[dict], max_tokens: int = 512) -> tuple[str, Usage]:
        body = {"model": self.name, "messages": messages, "max_tokens": max_tokens,
                "temperature": self.temperature}
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env) if self.api_key_env else None
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(self.base_url + "/chat/completions", method="POST",
                                     data=json.dumps(body).encode("utf-8"), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (408, 409, 429) or e.code >= 500:
                raise TransientError(f"HTTP {e.code} from {self.base_url}") from e
            raise PermanentError(f"HTTP {e.code} from {self.base_url}: {e.read()[:300]!r}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise TransientError(f"cannot reach {self.base_url}: {e}") from e
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise PermanentError(f"unexpected response shape: {str(data)[:300]}") from e
        u = data.get("usage") or {}
        pin, pout = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
        return text, Usage(pin, pout, (pin * self.usd_in + pout * self.usd_out) / 1e6)


Reply = Union[str, BaseException]


class ScriptedModel:
    """Deterministic stand-in for a model, for tests, demos and replaying failures.

    `script` is a list of replies (an exception in the list is raised instead), or a
    function that receives the messages and returns a reply.
    """

    def __init__(self, script: Sequence[Reply] | Callable[[list[dict]], str], *,
                 name: str = "scripted", usd_per_call: float = 0.0):
        self.script = script
        self.name = name
        self.usd_per_call = usd_per_call
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, messages: list[dict], max_tokens: int = 512) -> tuple[str, Usage]:
        if callable(self.script):
            with self._lock:
                self.calls += 1
            reply: Reply = self.script(messages)  # outside the lock so slow scripts run in parallel
        else:
            with self._lock:
                if self.calls >= len(self.script):
                    raise PermanentError(f"{self.name}: script has no reply #{self.calls + 1}")
                reply = self.script[self.calls]
                self.calls += 1
        if isinstance(reply, BaseException):
            raise reply
        prompt = sum(len(m.get("content", "")) for m in messages) // 4
        return reply, Usage(prompt, len(reply) // 4, self.usd_per_call)


class CachedModel:
    """Reuse a model's reply for an identical prompt: a repeat costs nothing and takes no time.

    sparkrun prompts are compact and stable (fixed system prompt, compact task view), so
    identical prompts recur across repeated tasks, retries and similar requests. Pass a dict
    to share one cache between models and runs, or a directory to keep it across processes.
    Only use it with deterministic models (temperature 0), where a repeat would say the same.
    """

    def __init__(self, model: Model, cache: "dict | str | os.PathLike | None" = None):
        self.model = model
        self.name = model.name
        self.dir = None if cache is None or isinstance(cache, dict) else Path(cache)
        self.cache: dict = cache if isinstance(cache, dict) else {}
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = 0
        self._lock = threading.Lock()

    def _key(self, messages: list[dict], max_tokens: int) -> str:
        data = json.dumps([self.name, messages, max_tokens], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def complete(self, messages: list[dict], max_tokens: int = 512) -> tuple[str, Usage]:
        key = self._key(messages, max_tokens)
        with self._lock:
            text = self.cache.get(key)
            if text is None and self.dir and (self.dir / key).exists():
                text = (self.dir / key).read_text(encoding="utf-8")
                self.cache[key] = text
            if text is not None:
                self.hits += 1
                return text, Usage(0, 0, 0.0)
        text, usage = self.model.complete(messages, max_tokens)  # outside the lock: calls run in parallel
        with self._lock:
            self.misses += 1
            self.cache[key] = text
            if self.dir:
                (self.dir / key).write_text(text, encoding="utf-8")
        return text, usage
