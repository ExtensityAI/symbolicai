# Symbolic Engine (WolframAlpha)

---

## ⚠️  Outdated or Deprecated Documentation ⚠️
This documentation is outdated and may not reflect the current state of the SymbolicAI library. This page might be revived or deleted entirely as we continue our development. We recommend using more modern tools that infer the documentation from the code itself, such as [DeepWiki](https://deepwiki.com/ExtensityAI/symbolicai). This will ensure you have the most accurate and up-to-date information and give you a better picture of the current state of the library.

---

Although our work primarily emphasizes how LLMs can assess symbolic expressions, many formal statements have already been efficiently implemented in existing symbolic engines, such as WolframAlpha. Therefore, with an API KEY from WolframAlpha, we can use their engine by using the `Interface('wolframalpha')`. This avoids error-prone evaluations from neuro-symbolic engines for mathematical operations. The following example demonstrates how to use WolframAlpha to compute the result of the variable `x`:

```python
from symai import Interface
expression = Interface('wolframalpha')
res = expression('x^2 + 2x + 1')
```

```bash
:Output:
x = -1
```
