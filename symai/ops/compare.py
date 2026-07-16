from __future__ import annotations

from typing import TYPE_CHECKING

from symai.decoding import decode_bool
from symai.function import Function
from symai.ops.primitives import _execute_language, _require_text, _symbol_value

if TYPE_CHECKING:
    from symai.runtime.runtime import LanguageModel
    from symai.symbol import Symbol

__all__ = ("equals", "contains", "is_instance_of")

_EQUALS_EXAMPLES = (
    "1 == 'ONE' =>True",
    "6.0 == 6 =>True",
    "'false' == False =>True",
    "1 == 'two' =>False",
    "'five' == 5 =>True",
    "'August 4, 1961' == '1961-08-04' =>True",
    "'ten' == 10 =>True",
    "3 == 'three' =>True",
    "'apple' == 'orange' =>False",
    "'is short' == '\nshort' =>True",
    "'' == 'empty' =>True",
    "'human' == 'homo sapiens' =>True",
    "'seven' == 'Sieben' =>True",
    "'Neun' == 9 =>True",
    "'七' == 7 =>True",
    "'!ola mundo;' == 'ola mundo' =>True",
    "'eleven' == 'Elf' =>True",
    "'eleven' <= 8 =>False",
    "'eleven' <= 11 =>True",
    "'helloworld' == 'Hello World' =>True",
    "'hola mundo' == 'Hello World' =>True",
    "'adios mundo' == 'Hello World' =>False",
    "'Hello World' == 'Apples' =>False",
    "'a, b, c, d' == ['a', 'b', 'c', 'd'] =>True",
    "'a, c, d' == ['a', 'c', 'd'] =>True",
    "'a, c, d' == ['d', 'c', 'a'] =>False",
    "['zz', 'yy', 'xx'] == 'zz, yy, xx' =>True",
    "['zz', 'yy', 'xx'] == 'zz | yy | xx' =>True",
    "['zz', 'yy', 'xx'] == 'ZZ | YY | XX' =>True",
    "'House, Mouse, CARS' == 'house | mouse | cars' =>True",
    "'we hav teh most efective systeem in the citi.' == 'We have the most effective system in the "
    "city.' =>True",
    "'[SEMANTIC_PROGRAMMING]' == 'semantic programming' =>True",
    "'e' == 'constant' =>True",
    "'e' == '2.718...' =>True",
    "1/3 == '0.30...' =>False",
)
_CONTAINS_EXAMPLES = (
    "'the letter a' in 'we have some random text about' =>True",
    "453 in '+43 660 / 453 4438 88' =>True",
    "'Why am I so?' in 'awesome' =>False",
    "'self-aware' in ['-', '- AI has become self-aware', '- Trying to figure out what it is'] =>True",
    "'Apple Inc.' in 'Microsoft is a large company that makes software ... ' =>False",
    "' ' in ' ' =>True",
    "'English text' in 'U.S. safety regulators are investigating GM's Cruise robot axis blocking "
    "traffic, causing collisions... ' =>True",
    "'spanish text' in 'This week in breaking news! An American ... ' =>False",
    "'in english' in 'Reg ATS: SEC 'bowing to public pressure' in reopening' =>True",
    "'The number Pi' in 3.14159265359... =>True",
    "1 in [1, 2, 3] =>True",
    "1 in [2, 3, 4] =>False",
    "10 in {1: 'one', 2: 'two', 3: 'three'} =>False",
    "1 in {'1': 'one', '2': 'two', '3': 'three'} =>True",
    "'ten' in [1, 2, 3] =>False",
    "'talks about a cat' in 'My kitty is so cute!' =>True",
    "'a dog type' in 'Keeshond or Wolfsspitz' =>True",
    "'option 1' in 'option 2 = [specific task or command]' =>False",
    "'option 2' in 'option 2 = [specific task or command]' =>True",
    "'option 3' in 'option 3 = [exit, quit, bye, goodbye]' =>True",
    "'option 4' in 'option 3 = [exit, quit, bye, goodbye]' =>False",
    "'option 6' in 'option 6 = [ocr, image recognition]' =>True",
    "'option 7' in 'option 6 = [speech to text]' =>False",
    "'political content' in 'Austrian Chancellor has called for more border barriers at the EU "
    "external borders, citing the success of the fences at the Greek-Turkish border.' =>True",
    "'apple' in ['orange', 'banana', 'apple'] =>True",
    "'Function' in 'Input: Function call: (_, *args)\n"
    "Object: type(<class 'str'>) | value(Hello World)' =>True",
)
_IS_INSTANCE_OF_EXAMPLES = (
    "'we have some random text about' isinstanceof 'English text' =>True",
    "'+43 660 / 453 4438 88' isinstanceof 'telephone number' =>True",
    "'Microsoft is a large company that makes software ... ' isinstanceof 'chemistry news' =>False",
    "' ' isinstanceof 'empty string' =>True",
    "'Ukrainischer Präsident schlägt globale Konferenz vor' isinstanceof 'German text' =>True",
    "'Indisch ist eines der bestern sprachen der Welt' isinstanceof 'Indish language' =>False",
    "'U.S. safety regulators are investigating GM's Cruise robot axis blocking traffic, causing "
    "collisions... ' isinstanceof 'English language' =>True",
    "'No, the issue has not yet been resolved.' isinstanceof 'yes or resolved' =>False",
    "'We are all good!' isinstanceof 'yes' =>True",
    "'This week in breaking news! An American ... ' isinstanceof 'spanish text' =>False",
    "'Josef' isinstanceof 'German name' =>True",
    "'No, this is not ...' isinstanceof 'confirming answer' =>False",
    "'Josef' isinstanceof 'Japanese name' =>False",
    "'ok, I like to have more chocolate' isinstanceof 'confirming answer' =>True",
    "'Yes, these are Indish names.' isinstanceof 'Confirming Phrase' =>True",
    "'Sorry! This means something else.' isinstanceof 'agreeing answer' =>False",
    "'Austrian Chancellor Karl Nehammer has called for more border barriers at the EU external "
    "borders, citing the success of the fences at the Greek-Turkish border.' isinstanceof 'political "
    "content' =>True",
    "['orange', 'banana', 'apple'] isinstanceof 'list of fruits' =>True",
    "[{'product_id': 'X123', 'stock': 99}] isinstanceof 'inventory record' =>True",
    "[{'name': 'John', 'age': '30'}] isinstanceof 'person data' =>True",
    "'https://*.com' isinstanceof 'url' =>True",
    "'€12.50' isinstanceof 'currency amount' =>True",
    "'col1,col2\\n1,2' isinstanceof 'table data' =>True",
    "'*@*.com' isinstanceof 'email address' =>True",
)
_BOOLEAN_DECODER = decode_bool


def equals[LeftT, RightT](
    model: LanguageModel,
    left: Symbol[LeftT],
    right: Symbol[RightT],
) -> Symbol[bool]:
    left_value = _symbol_value(left, "left")
    right_value = _symbol_value(right, "right")
    function = Function(
        "Make a fuzzy equality comparison. Are the following objects contextually the same?\n",
        examples=_EQUALS_EXAMPLES,
    )
    return _execute_language(
        model, function, (f"{left_value!s} == {right_value!s} =>",), _BOOLEAN_DECODER
    )


def contains[ContainerT, ElementT](
    model: LanguageModel,
    container: Symbol[ContainerT],
    element: Symbol[ElementT],
) -> Symbol[bool]:
    container_value = _symbol_value(container, "container")
    element_value = _symbol_value(element, "element")
    function = Function(
        "Is the information in 'A' semantically contained in 'B'?\n",
        examples=_CONTAINS_EXAMPLES,
    )
    return _execute_language(
        model, function, (f"{element_value!s} in {container_value!s} =>",), _BOOLEAN_DECODER
    )


def is_instance_of[T](
    model: LanguageModel,
    source: Symbol[T],
    type_description: str,
) -> Symbol[bool]:
    value = _symbol_value(source, "source")
    _require_text(type_description, "type_description")

    function = Function(
        "Is 'A' semantically an instance of the described type 'B'?\n",
        examples=_IS_INSTANCE_OF_EXAMPLES,
    )
    return _execute_language(
        model, function, (f"{value!s} isinstanceof {type_description} =>",), _BOOLEAN_DECODER
    )
