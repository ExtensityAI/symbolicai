# Search Engine

To obtain fact-based content, we can perform search queries via `SerpApi` with a `Google` backend. The following example demonstrates how to search for a query and return the results:

```python
from symai.interfaces import Interface

search = Interface('serpapi')
res = search('Birthday of Barack Obama')
```

```bash
:Output:
August 4, 1961
```