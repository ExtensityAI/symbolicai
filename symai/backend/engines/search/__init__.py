from symai.backend.engines.search.engine_firecrawl import FirecrawlEngine
from symai.backend.engines.search.engine_parallel import ParallelEngine

SEARCH_ENGINE_MAPPING = {
    "firecrawl": FirecrawlEngine,
    "parallel": ParallelEngine,
}

__all__ = [
    "SEARCH_ENGINE_MAPPING",
    "FirecrawlEngine",
    "ParallelEngine",
]
