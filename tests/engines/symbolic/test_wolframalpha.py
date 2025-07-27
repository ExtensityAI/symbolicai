from symai.extended import Interface
from symai.utils import semassert

try:
    import wolframalpha
except ImportError:
    raise ImportError("wolframalpha is not installed. Please install it using 'pip install wolframalpha'")

symbolic = Interface('wolframalpha')

def test_wolframalpha():
    res = symbolic('What is the capital of France?')
    semassert('Paris' in res.value, f"Expected 'Paris' in '{res.value}'")

def test_simple_arithmetic():
    res = symbolic('What is 2+2?')
    semassert('4' in res.value, f"Expected '4' in '{res.value}'")

def test_math_derivative():
    res = symbolic('What is the derivative of sin(x)?')
    semassert('cos(x)' in res.value or 'cos' in res.value, f"Expected 'cos(x)' in '{res.value}'")

def test_math_integral():
    res = symbolic('What is the integral of x^2 from 0 to 3?')
    semassert('9' in res.value, f"Expected '9' in '{res.value}'")

def test_solve_quadratic():
    res = symbolic('Solve x^2 - 4 = 0 for x')
    semassert('± 2' in res.value, f"Expected '-2' and '2' in '{res.value}'")
