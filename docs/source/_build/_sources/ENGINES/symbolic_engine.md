# Symbolic Engine (WolframAlpha)

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