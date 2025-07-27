# Symbolic Engine (WolframAlpha)

 > ❗️**NOTE**❗️ As of July 27, 2025, WolframAlpha has a very well known issue that hasn't been fixed yet. There is a manual fix that can be found [here](https://github.com/jaraco/wolframalpha/pull/34/commits/6eb3828ee812f65592e00629710fc027d40e7bd1)

Although our work primarily emphasizes how LLMs can assess symbolic expressions, many formal statements have already been efficiently implemented in existing symbolic engines, such as WolframAlpha. Therefore, with an API KEY from WolframAlpha, we can use their engine by using the `Interface('wolframalpha')`. This avoids error-prone evaluations from neuro-symbolic engines for mathematical operations. The following example demonstrates how to use WolframAlpha to compute the result of the variable `x`:

```python
from symai import Interface
expression = Interface('wolframalpha')
res = expression('x^2 + 2x + 1')
# x = -1
```
