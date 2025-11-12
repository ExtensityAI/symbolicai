# Scrape Engine

To access data from the web, we can use the `naive_scrape` interface. The engine underneath is very lightweight and can be used to scrape data from websites. It is based on the `requests` library, as well as `trafilatura` for output formatting, and `bs4` for HTML parsing. `trafilatura` currently supports the following output formats: `json`, `csv`, `html`, `markdown`, `text`, `xml`

```python
from symai.interfaces import Interface

scraper = Interface("naive_scrape")
url = "https://docs.astral.sh/uv/guides/scripts/#next-steps"
res = scraper(url)
```
