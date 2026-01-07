from typing import TYPE_CHECKING

from ...backend.engines.index.engine_qdrant import QdrantIndexEngine
from ...symbol import Expression, Symbol

if TYPE_CHECKING:
    from ...backend.engines.index.engine_qdrant import SearchResult


class local_search(Expression):
    def __init__(self, index_name: str = QdrantIndexEngine._default_index_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index_name = index_name
        self.name = self.__class__.__name__

    def search(self, query: Symbol, **kwargs) -> "SearchResult":
        symbol = self._to_symbol(query)
        options = dict(kwargs)

        index_name = options.pop("collection_name", options.pop("index_name", self.index_name))

        # Normalize limit/top_k/index_top_k
        index_top_k = options.pop("index_top_k", None)
        if index_top_k is None:
            top_k = options.pop("top_k", None)
            limit = options.pop("limit", None)
            index_top_k = top_k if top_k is not None else limit
        if index_top_k is not None:
            options["index_top_k"] = index_top_k

        # Bypass decorator/EngineRepository pipeline entirely (and thus `forward()`).
        # We query Qdrant directly and then format results into the same SearchResult
        # structure used by `parallel.search` (citations, inline markers, etc.).
        engine = QdrantIndexEngine(index_name=index_name)
        try:
            score_threshold = options.pop("score_threshold", None)
            raw_filter = options.pop("query_filter", options.pop("filter", None))
            query_filter = engine._build_query_filter(raw_filter)

            # Keep `with_payload` default aligned with engine behavior; let caller override.
            with_payload = options.pop("with_payload", True)
            with_vectors = options.pop("with_vectors", False)

            points = engine._search_sync(
                collection_name=index_name,
                query_vector=symbol.embedding,
                limit=options.pop("index_top_k", engine.index_top_k),
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=with_payload,
                with_vectors=with_vectors,
                **options,
            )
            result = engine._format_search_results(points, index_name)
        finally:
            del engine
        return result
