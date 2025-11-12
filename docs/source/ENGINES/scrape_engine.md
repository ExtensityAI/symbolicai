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
