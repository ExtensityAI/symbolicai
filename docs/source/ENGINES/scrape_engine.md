# Scrape Engine

## Naive Scrape

To access data from the web, we can use the `naive_scrape` interface. The engine underneath is very lightweight and can be used to scrape data from websites. It is based on the `requests` library, as well as `trafilatura` for output formatting, and `bs4` for HTML parsing. `trafilatura` currently supports the following output formats: `json`, `csv`, `html`, `markdown`, `text`, `xml`

```python
from symai.interfaces import Interface

scraper = Interface("naive_scrape")
url = "https://docs.astral.sh/uv/guides/scripts/#next-steps"
res = scraper(url)
```

## Parallel (Parallel.ai)

The Parallel.ai integration routes scrape calls through the official `parallel-web` SDK and can handle PDFs, JavaScript-heavy feeds, and standard HTML pages in the same workflow. Instantiate the Parallel interface and call `.scrape(...)` with the target URL. The engine detects scrape requests automatically whenever a URL is supplied.

```python
from symai.extended import Interface

scraper = Interface("parallel")
article = scraper.scrape(
    "https://trafilatura.readthedocs.io/en/latest/crawls.html",
    full_content=True,           # optional: request full document text
    excerpts=True,               # optional: default True, retain excerpt snippets
    objective="Summarize crawl guidance for internal notes."
)
print(str(article))
```

Configuration requires a Parallel API key and the Parallel model token. Add the following to your settings:

```bash
{
    …
    "SEARCH_ENGINE_API_KEY": "…",
    "SEARCH_ENGINE_MODEL": "parallel"
    …
}
```

When invoked with a URL, the engine hits Parallel's Extract API and returns an `ExtractResult`. The result string joins excerpts or the full content if requested. Because processing is offloaded to Parallel's hosted infrastructure, the engine remains reliable on dynamic pages that the naive scraper cannot render. Install the dependency with `pip install parallel-web` before enabling this engine.

## Firecrawl

Firecrawl.dev specializes in reliable web scraping with automatic handling of JavaScript-rendered content, proxies, and anti-bot mechanisms. It converts web pages into clean formats suitable for LLM consumption and supports advanced features like actions, caching, and location-based scraping.

### Examples

```python
from symai.extended import Interface

scraper = Interface("firecrawl")

# Example 1: Basic webpage scraping
content = scraper.scrape(
    "https://docs.firecrawl.dev/introduction",
    formats=["markdown"]
)
print(content)

# Example 2: PDF scraping with content extraction and trimming
pdf_full = scraper.scrape(
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7231600",
    only_main_content=True,
    formats=["markdown"],
    proxy="auto"
)
# Trim locally if needed
pdf_trimmed = str(pdf_full)[:100]

# Note: JS-heavy sites like Twitter/LinkedIn are currently not fully supported
# They typically return 403 Forbidden errors (may vary by subscription tier)
```

### Configuration

Enable the engine by configuring Firecrawl credentials:

```bash
{
    "SEARCH_ENGINE_API_KEY": "fc-your-api-key",
    "SEARCH_ENGINE_MODEL": "firecrawl"
}
```

### JSON Schema Extraction

Firecrawl supports structured data extraction using JSON schemas. This is useful for extracting specific fields from web pages using LLM-powered extraction:

```python
from pydantic import Field
from symai.extended import Interface
from symai.models import LLMDataModel

class MetadataModel(LLMDataModel):
    """Bibliographic metadata extracted from a source document."""
    title: str = Field(description="Title of the source.")
    year: str = Field(description="Publication year (4 digits) or Unknown.")
    authors: list[str] = Field(default_factory=list, description="List of authors.")
    doi: str | None = Field(default=None, description="DOI if available.")

# Build JSON format config from Pydantic schema
schema = MetadataModel.model_json_schema()
json_format = {
    "type": "json",
    "prompt": "Extract bibliographic metadata from this academic paper.",
    "schema": schema,
}

scraper = Interface("firecrawl")
result = scraper.scrape(
    "https://journals.physiology.org/doi/full/10.1152/ajpregu.00051.2002",
    formats=[json_format],
    proxy="auto"
)

# Access extracted data as dict
extracted = result.raw["json"]
metadata = MetadataModel(**extracted)
print(metadata.model_dump())

# Or as JSON string
print(str(result))
```

### Supported Parameters

The engine supports many parameters (passed as kwargs). Common ones include:
- **formats**: Output formats (["markdown"], ["html"], ["rawHtml"])
- **only_main_content**: Extract main content only (boolean)
- **proxy**: Proxy mode ("basic", "stealth", "auto")
- **location**: Geographic location object with optional country and languages
  - Example: `{"country": "US"}` or `{"country": "RO", "languages": ["ro"]}`
- **maxAge**: Cache duration in seconds (integer)
- **storeInCache**: Enable caching (boolean)
- **actions**: Page interactions before scraping (list of action objects)
  - Example: `[{"type": "wait", "milliseconds": 2000}]`
  - Example: `[{"type": "click", "selector": ".button"}]`
  - Example: `[{"type": "scroll", "direction": "down", "amount": 500}]`

Check the Firecrawl v2 API documentation for the complete list of available parameters and action types.
