from .engine_firecrawl import FirecrawlEngine
from .engine_parallel import ParallelEngine

SEARCH_ENGINE_MAPPING = {
    "firecrawl": FirecrawlEngine,
    "parallel": ParallelEngine,
}

__all__ = [
    "SEARCH_ENGINE_MAPPING",
    "FirecrawlEngine",
    "ParallelEngine",
]
