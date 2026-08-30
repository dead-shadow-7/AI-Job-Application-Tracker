"""Local text embeddings.

Runs BAAI/bge-small-en-v1.5 through fastembed (ONNX, CPU, no PyTorch). Chosen
over a hosted embedding API for three reasons, in order of importance:

1. **A resume never leaves the machine.** It is the most personal document in
   this system, and not sending it anywhere is simpler than any policy about
   how a third party may use it.
2. No per-call cost and no rate limit, so re-embedding every resume after a
   chunking change is free rather than a budgeting decision.
3. Neither Groq nor most chat providers serve embeddings at all, so a hosted
   option would mean a second vendor regardless.

The trade-off is a ~130 MB model download on first use and CPU inference. For
one person's resume and job list that is milliseconds per chunk.
"""

import asyncio
import logging
from typing import Protocol

from starlette.concurrency import run_in_threadpool

from app.core.config import settings

logger = logging.getLogger(__name__)

# bge models are trained with an asymmetric prefix: queries get one, documents
# do not. Omitting it costs a few points of retrieval quality, and the effect is
# invisible — results are merely slightly worse, never obviously wrong.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingProvider(Protocol):
    dimension: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


class FastEmbedProvider:
    """Lazily-loaded local embedder.

    The model is loaded on first use rather than at import: it downloads on
    first run, and paying that during application startup would make the API
    appear hung. A lock guards the load so concurrent first requests do not each
    start their own download.
    """

    def __init__(self, model_name: str | None = None, dimension: int | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self.dimension = dimension or settings.embedding_dim
        self._model = None
        self._lock = asyncio.Lock()

    async def _ensure_model(self):  # type: ignore[no-untyped-def]
        if self._model is not None:
            return self._model

        async with self._lock:
            if self._model is not None:  # another task won the race
                return self._model

            from fastembed import TextEmbedding

            logger.info("Loading embedding model %s (first use may download)", self.model_name)
            self._model = await run_in_threadpool(TextEmbedding, model_name=self.model_name)
            logger.info("Embedding model ready")
        return self._model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = await self._ensure_model()
        # fastembed is synchronous and CPU-bound; a threadpool keeps it off the
        # event loop so concurrent requests are not blocked by one upload.
        vectors = await run_in_threadpool(lambda: list(model.embed(texts)))
        return [v.tolist() for v in vectors]

    async def embed_query(self, text: str) -> list[float]:
        model = await self._ensure_model()
        prefixed = f"{QUERY_PREFIX}{text}"
        vectors = await run_in_threadpool(lambda: list(model.query_embed(prefixed)))
        return vectors[0].tolist()


embedding_provider: EmbeddingProvider = FastEmbedProvider()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Only for comparing two vectors already in memory.

    Anything involving a *search* belongs in Postgres, where the HNSW index
    does the work; pulling rows out to compare them here would defeat it.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
