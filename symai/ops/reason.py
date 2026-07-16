from __future__ import annotations

from typing import TYPE_CHECKING

from symai.decoding import TextDecoder
from symai.function import Function
from symai.ops.primitives import _execute_language
from symai.symbol import Symbol

if TYPE_CHECKING:
    from symai.runtime.runtime import LanguageModel

__all__ = ("query", "interpret", "logic")

_INTERPRET_EXAMPLES = (
    "doctor - male + female =>nurse",
    "Paris - France + Italy =>Rome",
    "hot - summer + winter =>cold",
    "lion - adult + young =>cub",
    "teacher - school + hospital =>doctor",
    '"Lanterns shimmer beside the river" + "Fireflies sketch constellations in the dark" =>Lanterns '
    "shimmer beside the river while fireflies sketch constellations in the dark.",
    '"Rain drums gently on the roof" - "gently" =>Rain drums on the roof.',
    '"Leaves twirl across the pavement" * "Waves hush the midnight shore" =>Nature twirls and hushes '
    "across pavement and shore.",
    '"The bakery smells of cinnamon" / "Morning begins" =>If morning begins, the bakery smells of '
    "cinnamon.",
    'not("The sky glows crimson at dusk") =>The sky does not glow crimson at dusk.',
    '"Birds greet dawn with song" and "The library hums with whispers" =>Birds greet dawn with song '
    "and the library hums with whispers.",
    '"A lone cat prowls the alley" or "Leaves twirl across the pavement" =>Either a lone cat prowls '
    "the alley or leaves twirl across the pavement.",
    '"The campfire crackles and sparks" xor "Rain drums on the roof" =>Either the campfire crackles '
    "and sparks or rain drums on the roof, but not both.",
    '"The sky glows crimson at dusk" implies "Night soon follows" =>If the sky glows crimson at dusk, '
    "then night soon follows.",
    '"Fireflies sketch constellations in the dark" ++ "Lanterns shimmer beside the river" =>A '
    "festival of lights sparkles against the night by the river.",
    '"Rain drums on the roof" >> "Sleep comes easily" =>Rain drums on the roof, so sleep comes '
    "easily.",
    '"Birds greet dawn with song" || "Lanterns fade in the river breeze" =>One scene wakes while the '
    "other fades.",
    '"Waves hush the midnight shore" + "The campfire crackles and sparks" - "midnight" =>Waves hush '
    "the shore while the campfire crackles and sparks.",
    '"The violinist fills the plaza with melody" * "Birds greet dawn with song" =>Music ripples '
    "through dawn as birds and violinist weave a shared melody.",
    '"x + y = 10" + "y = 3" =>x + y = 10 and y = 3.',
    '"x + y = 10" / "y = 3" =>If y = 3, then x + 3 = 10.',
    '"2x = 8" >> "x = 4" =>Because 2x = 8, x = 4.',
    'not("x = 5") =>x ≠ 5.',
    '"x² = 9" or "x = 4" =>Either x² = 9 or x = 4.',
    '"x² = 9" xor "x = 4" =>Exactly one of x² = 9 or x = 4, but not both.',
    '"x² = 9" implies "x = ±3" =>If x² = 9, then x = ±3.',
    '"f prime (x) = 0" ++ "f has a local extremum" =>A critical point indicates f has a local '
    "extremum.",
    '"a² + b² = c²" * "c = 13" =>In the right-triangle where c = 13, a² + b² = 169.',
    '"limₓ→0 sin x / x = 1" and "x approaches 0" =>As x approaches 0, sin x / x tends to 1.',
    '"SELECT name FROM customers" + "WHERE city = \'Paris\'" =>SELECT name FROM customers WHERE city '
    "= 'Paris'.",
    '"for i in range(5): print(i)" - "print(i)" =>for i in range(5):',
    "\"def greet(name): return 'Hi ' + name\" >> \"greet('Leo')\" =>Because we define greet, "
    "greet('Leo').",
    '"x > 3" and "x < 7" =>3 < x < 7.',
    '"a divides b" implies "b mod a = 0" =>If a divides b, then b mod a = 0.',
    '"p" xor "not p" =>Exactly one of p or not p, but not both.',
    '"f prime (x) exists" ++ "f(x) continuous" =>A differentiable function is necessarily continuous.',
    "\"SELECT * FROM orders\" / \"status = 'PENDING'\" =>If status = 'PENDING', SELECT * FROM orders.",
    '"x = 2" + "y = 3" * "z = x + y" =>With x = 2 and y = 3, z = 5.',
    '"temperature rises" >> "ice melts" =>Because temperature rises, ice melts.',
)
_LOGIC_EXAMPLES = (
    "expr True and True =>'True'",
    "expr True and False =>'False'",
    "expr False and True =>'False'",
    "expr False and False =>'False'",
    "expr True or True =>'True'",
    "expr True or False =>'True'",
    "expr False or True =>'True'",
    "expr False or False =>'False'",
    "expr True xor True =>'False'",
    "expr True xor False =>'True'",
    "expr False xor True =>'True'",
    "expr False xor False =>'False'",
    "expr 'All humans are mortal' and 'Socrates is a human' =>'Therefore, Socrates is mortal.'",
    "expr 'If it rains, the ground gets wet' and 'It is raining' =>'Therefore, the ground gets wet.'",
    "expr 'The sky is blue' and 'The sky is not blue' =>'Contradiction - both cannot be true "
    "together.'",
    "expr 'It is Monday' or 'It is a holiday' =>'Either it is Monday, a holiday, or possibly both.'",
    "expr 'Alice is at home' or 'Bob is at home' =>'Alice or Bob is at home, perhaps both.'",
    "expr 'The light is red' xor 'The light is green' =>'The light is either red or green, but not "
    "both.'",
    "expr 'She won the prize' xor 'He won the prize' =>'Either she or he won the prize, but not "
    "both.'",
    "expr 'The engine is running' xor 'The engine is not running' =>'Either the engine is running or "
    "it is not, but not both.'",
)


def query[T](
    model: LanguageModel,
    source: Symbol[T],
    question: str,
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    _require_text(question, "question")
    function = Function("Answer the question using only the provided data:\n")
    return _execute_language(
        model, function, (f"Data:\n{value!s}\nQuestion: {question}\nAnswer:",), TextDecoder()
    )


def interpret[T](
    model: LanguageModel,
    source: Symbol[T],
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    function = Function(
        "Evaluate the symbolic expression and return only the result:\n",
        examples=_INTERPRET_EXAMPLES,
    )
    return _execute_language(model, function, (f"{value!s} =>",), TextDecoder())


def logic[LeftT, RightT](
    model: LanguageModel,
    left: Symbol[LeftT],
    operator: str,
    right: Symbol[RightT],
) -> Symbol[str]:
    left_value = _symbol_value(left, "left")
    right_value = _symbol_value(right, "right")
    _require_text(operator, "operator")
    function = Function(
        "Evaluate the logic expression:\n",
        examples=_LOGIC_EXAMPLES,
    )
    return _execute_language(
        model, function, (f"expr {left_value!s} {operator} {right_value!s} =>",), TextDecoder()
    )


def _symbol_value[T](symbol: Symbol[T], field: str) -> T:
    if not isinstance(symbol, Symbol):
        msg = f"{field} must be a Symbol"
        raise TypeError(msg)

    return symbol.value


def _require_text(value: object, field: str) -> None:
    if not isinstance(value, str):
        msg = f"{field} must be text"
        raise TypeError(msg)
