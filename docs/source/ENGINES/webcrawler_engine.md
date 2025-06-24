# Webcrawler Engine

---

## ⚠️  Outdated or Deprecated Documentation ⚠️
This documentation is outdated and may not reflect the current state of the SymbolicAI library. This page might be revived or deleted entirely as we continue our development. We recommend using more modern tools that infer the documentation from the code itself, such as [DeepWiki](https://deepwiki.com/ExtensityAI/symbolicai). This will ensure you have the most accurate and up-to-date informatio and give you a better picture of the current state of the library.

---
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
