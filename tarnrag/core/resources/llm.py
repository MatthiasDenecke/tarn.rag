"""The LanguageModel seam — a ``core`` ``Resource`` (like the embedder / cross-encoder), provider-pluggable.

The reader/decomposer LLM lives in ``core/`` (not ``generation/``) so **any** layer that opts in can inject
it: the generation ``Reasoner`` (its heavy user) **and** an optional LLM-based retrieval classifier/router.
A retrieval component using it depends on ``core`` (``retrieval → core``), so the one-way
``generation → retrieval`` rule is untouched; retrieval's *default* path uses no LLM.

A ``LanguageModel`` is a ``Resource`` (engine-built, injected at call time via a context), **not** a
``Component``: provider-pluggable behind a selector, lazily-loaded client, faked in tests by injecting a
``StaticLanguageModel``. ``complete`` is async — the API call is awaited (and the C++ port can implement it
too: an LLM call is just HTTP, not a portability barrier).
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tarnrag.core.resources.resource import Resource

if TYPE_CHECKING:
    from tarnrag.core.engine.config import LLMSettings


@dataclass(frozen=True)
class Prompt:
    """A single-turn completion request: an optional ``system`` instruction + the ``user`` text, plus the
    sampling knobs. ``max_tokens`` / ``temperature`` default to ``None`` — "no per-call preference" — so
    the backend falls back to its configured ``LLMSettings`` defaults (the three-level fallback lives in
    ``llm_api._sampling``). Kept minimal and provider-agnostic; each backend maps it to its own request
    shape."""

    user: str
    system: str | None = None
    max_tokens: int | None = None  # None → the configured LLMSettings default (see llm_api._sampling)
    temperature: float | None = None  # None → the configured LLMSettings default


@dataclass(frozen=True)
class Completion:
    """A model's reply: the generated ``text`` plus optional metadata (stop reason, token usage)."""

    text: str
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


class LanguageModel(Resource):
    """Port: complete a ``Prompt`` to a ``Completion`` — the reader the generation ``Reasoner`` consumes
    (and an optional LLM-based retrieval classifier). Provider-pluggable; injected, not registered."""

    @abstractmethod
    async def complete(self, prompt: Prompt) -> Completion:
        """Generate a completion for ``prompt``."""

    async def aclose(self) -> None:
        """Release any lazily-built network client. Default: nothing to release (offline backends);
        the HTTP/SDK backends override it. Idempotent — whoever *built* the model closes it."""

    @staticmethod
    def create(llm: LLMSettings) -> LanguageModel:
        """Build the ``LanguageModel`` for the configured provider, mapping the ``LLMSettings`` slice to
        the provider's constructor. The provider map is local (single use) and the import lazy (it pulls
        the optional SDK). LLM construction is uniform, so — unlike ``Embedder.create`` — no per-provider
        ``create`` is needed; a provider with bespoke construction would reintroduce one."""
        from tarnrag.core.resources.llm_api import (
            AnthropicLanguageModel,
            OpenAILanguageModel,
        )

        providers: dict[str, type[LanguageModel]] = {
            "anthropic": AnthropicLanguageModel,
            "openai": OpenAILanguageModel,  # OpenAI itself or any compatible endpoint (vLLM/Together/…)
        }
        provider = providers.get(llm.provider)
        if provider is None:
            raise ValueError(f"unknown LLM provider {llm.provider!r}; choose from {sorted(providers)}")
        return provider(
            model=llm.model,
            api_key=llm.resolved_api_key(),  # explicit key, else the env var named by llm.api_key_env
            base_url=llm.api_base_url,
            max_tokens=llm.max_tokens,
            temperature=llm.temperature,
            timeout=llm.api_timeout,
        )


class StaticLanguageModel(LanguageModel):
    """A deterministic, network-free ``LanguageModel``: returns a canned reply, or one computed from the
    prompt by a callable. The test double *and* an offline/demo backend — the LLM analog of a fake
    embedder, but shipped, so generation can run with no API key."""

    def __init__(self, reply: str | Callable[[Prompt], str] = "", *, name: str = "static") -> None:
        self._reply = reply
        self._name = name

    def identity(self) -> str:
        return f"static:{self._name}"

    async def complete(self, prompt: Prompt) -> Completion:
        text = self._reply(prompt) if callable(self._reply) else self._reply
        return Completion(text=text, stop_reason="static")
