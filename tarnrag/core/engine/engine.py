"""Base class for the engine facades — the shared async lifecycle over the repository.

``IngestionEngine`` (the index producer) and ``RetrievalEngine`` (the reader) both run over a single
``DocumentRepository``; each builds it (and the embedder) from ``Settings`` via the resource factories
(``DocumentRepository.create`` / ``Embedder.create``, both passed ``EMBEDDING_DIMENSION`` so the index
and the query embedder agree on vector size). ``Engine`` holds what remains shared: the async-context
lifecycle (``aclose`` + the context-manager protocol) over that repository.
"""

from __future__ import annotations

from typing import Self

from tarnrag.storage.repository import DocumentRepository


class Engine:
    """
    Shared base for the two engine facades. Subclasses construct via their own ``create`` /
    ``open`` (which set ``repository``); the base supplies the repository lifecycle (``aclose`` +
    async-context manager).

    Ownership: an engine releases only what it built. ``create()`` that builds its own store sets
    ``owns_repository``; a store passed in (by a composition root like ``TarnRag``, an eval harness, a
    test) belongs to whoever passed it, so closing one engine never pulls the store from under the others.
    """

    repository: DocumentRepository  # set by each subclass's constructor
    owns_repository: bool = False  # True only when this engine's create() built the repository

    async def aclose(self) -> None:
        """Release the repository (its async engine pool) — when this engine built it."""
        if self.owns_repository:
            await self.repository.disconnect()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
