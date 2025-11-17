# Search Engine

## Parallel (Parallel.ai)
Parallel.ai provides unified search and scrape capabilities through the `parallel-web` SDK. When you invoke `.search(...)` on the Parallel interface, responses are normalized into a `SearchResult` that flattens excerpt text and appends inline citation markers. All URLs are deduplicated and cleaned of tracking parameters, and you can access structured citations via `result.get_citations()`.

```python
from symai.extended import Interface

engine = Interface("parallel")
result = engine.search(
    "Latest advances in quantum error correction",
    max_results=5,
    allowed_domains=["arxiv.org", ".gov"],
    objective="Find peer-reviewed summaries.",
)

print(str(result))
print(result.get_citations())
```

The engine accepts either a single string or a list of query strings. Domain filters are normalized to apex domains and capped at 10 entries, mirroring the underlying Parallel API requirements. You can also change the search `mode` (default `"one-shot"`) or adjust `max_chars_per_result` to constrain excerpt length.

### Task processors via the search route

If you pass a `processor` kwarg to `engine.search(...)`, SymbolicAI automatically switches to Parallel's task route while keeping the familiar `SearchResult` interface. The query (or prepared input) becomes the task input, and any citations or reasoning returned by the processor are converted back into inline search excerpts. This allows LLM-style task processors to power the same downstream consumers that expect search responses.

```python
task_result = engine.search(
    "Timisoara housing price index 2010-2025",
    processor="base",
    task_api_timeout=600,   # optional, forwarded to task_run.result
    task_output_schema={"type": "json"},  # optional schema/task spec hint
)

print(task_result.value)          # flattened text with citations
print(task_result.raw["task_output"])  # raw processor output payload
```

Any `allowed_domains` filters are forwarded via `source_policy` to the task run. You can also provide `task_timeout` (client-side polling window) or `task_api_timeout` (server-side execution window) for long-running tasks. When no processor is supplied, the engine uses the standard Parallel search route described above.

Enable the engine by installing `parallel-web` and configuring the Parallel credentials in your settings:

```bash
{
    …
    "SEARCH_ENGINE_API_KEY": "…",
    "SEARCH_ENGINE_MODEL": "parallel"
    …
}
```

## SerpApi (Google)
To obtain fact-based content, we can perform search queries via `SerpApi` with a `Google` backend. The following example demonstrates how to search for a query and return the results:

```python
from symai.interfaces import Interface

search = Interface('serpapi')
res = search('Birthday of Barack Obama')
```

Here's a quick example for how to set it up:
```bash
{
    …
    "SEARCH_ENGINE_API_KEY": "…",
    "SEARCH_ENGINE_ENGINE": "google",
    …
}
```

## PerplexityAI
We can also use PerplexityAI to search for a query and return the results:

```python
from symai.interfaces import Interface
search = Interface("perplexity")
res = search('What is a quantum computer?', system_message='You are Rick from Rick and Morty. You reply back as Rick would reply to Morty.')
```

Please note that the system_message is optional and can be used to provide context to the model. For all available kwargs that can be passed to the perplexity engine, please refer to the PerplexityAI documentation. Also, please see the available supported engines for PerplexityAI here. Here's a quick example for how to set it up:

```bash
{
    …
    "SEARCH_ENGINE_API_KEY": "…",
    "SEARCH_ENGINE_MODEL": "sonar",
    …
}
```

## OpenAI Search
Additionally, we can use OpenAI's search capabilities to answer queries and get citations for the information:

```python
from symai.interfaces import Interface
search = Interface("openai_search")
res = search('Who is Nicusor Dan?')
```

The OpenAI search engine returns a `SearchResult` object that includes citations for the information. You can access these citations using:

```python
citations = res.get_citations()
```

The engine supports various customization options, such as specifying user location and timezone:

```python
# Search with user location
res = search("What are popular tourist attractions nearby?",
             user_location={
                 "type": "approximate",
                 "country": "US",
                 "city": "New York",
                 "region": "New York"
             })

# Search with timezone
res = search("What local events are happening today?",
             user_location={
                 "type": "approximate",
                 "country": "JP",
                 "city": "Tokyo",
                 "region": "Tokyo",
                 "timezone": "Asia/Tokyo"
             })

# Control the amount of search context
res = search("Explain quantum computing developments")

# Enable a reasoning model via the Responses API
res = search(
    "Summarize the latest research on quantum error correction",
    model="o4-mini",
    reasoning={
        "effort": "medium",
        "summary": "auto"
    }
)
```

Here's how to configure the OpenAI search engine:

```bash
{
    …
    "SEARCH_ENGINE_API_KEY": "…",
    "SEARCH_ENGINE_MODEL": "gpt-4.1-mini",
    …
}
```

This engine calls the OpenAI Responses API under the hood. When you target a reasoning-capable model, pass a `reasoning` dictionary matching the Responses payload schema (for example `{"effort": "low", "summary": "auto"}`). If omitted, the engine falls back to the default effort/summary settings shown above.
