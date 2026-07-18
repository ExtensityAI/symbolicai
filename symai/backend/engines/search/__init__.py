from symai.backend.engines.search.firecrawl import FirecrawlEngine
from symai.backend.engines.search.gemini import GeminiSearchEngine
from symai.backend.engines.search.openai import GPTXSearchEngine
from symai.backend.engines.search.parallel import ParallelEngine
from symai.backend.engines.search.perplexity import PerplexityEngine

SEARCH_ENGINE_MAPPING = {
    "firecrawl": FirecrawlEngine,
    "parallel": ParallelEngine,
}

__all__ = [
    "SEARCH_ENGINE_MAPPING",
    "FirecrawlEngine",
    "GPTXSearchEngine",
    "GeminiSearchEngine",
    "ParallelEngine",
    "PerplexityEngine",
]
