# Webcrawler Engine

To access data from the web, we can use `Selenium`. The following example demonstrates how to crawl a website and return the results:

```python
from symai.interfaces import Interface

crawler = Interface('selenium')
res = crawler(url="https://www.google.com/",
              pattern="google")
```
The `pattern` property can be used to verify if the document has been loaded correctly. If the pattern is not found, the crawler will timeout and return an empty result.

```bash
:Output:
GoogleKlicke hier, wenn du nach einigen Sekunden nicht automatisch weitergeleitet wirst.GmailBilderAnmelden ...
```