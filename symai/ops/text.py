from __future__ import annotations

from typing import TYPE_CHECKING

from symai.decoding import decode_text
from symai.function import Function
from symai.ops.primitives import _execute_language, _require_text, _symbol_value
from symai.symbol import Symbol

if TYPE_CHECKING:
    from symai.runtime.runtime import LanguageModel

__all__ = (
    "summarize",
    "translate",
    "modify",
    "filter",
    "map",
    "convert",
    "style",
    "template",
    "replace",
    "include",
    "combine",
    "extract",
)

_MODIFY_EXAMPLES = (
    "text 'The quick brown fox jumps over the lazy dog.' modify 'fox to hours' =>The quick brown "
    "hours jumps over the lazy dog.",
    "text 'My cats name is Pucki' modify 'all caps' =>MY CATS NAME IS PUCKI",
    "text 'The square root of pi is 1.77245...' modify 'text to latex formula' "
    "=>$\\sqrt[2]{\\pi}=1.77245\\dots$",
    "text 'I hate this fucking product so much, because it lag's all the time.' modify 'curse words "
    "with neutral formulation' =>I hate this product since it lag's all the time.",
    "text 'Hi, whats up? Our new products is awesome with a blasting set of features.' modify "
    "'improve politeness and text quality' =>Dear Sir or Madam, I hope you are doing well. Let me "
    "introduce our new products with a fantastic set of new features.",
    "text 'Microsoft release a new chat bot API to enable human to machine translation.' modify "
    "'language to German' =>Microsoft veröffentlicht eine neue Chat-Bot-API, um die Übersetzung von "
    "Mensch zu Maschine zu ermöglichen.",
    "text '{\n"
    '    "name": "Manual Game",\n'
    '    "type": "python",\n'
    '    "request": "launch",\n'
    '    "program": "${workspaceFolder}/envs/textgrid.py",\n'
    '    "cwd": "${workspaceFolder}",\n'
    '    "args": [\n'
    '        "--debug"\n'
    "    ],\n"
    '    "env": {\n'
    '        "PYTHONPATH": "."\n'
    "    }\n"
    "}' modify 'json to yaml' =>name: Manual Game\n"
    "type: python\n"
    "request: launch\n"
    "program: ${workspaceFolder}/envs/textgrid.py\n"
    "cwd: ${workspaceFolder}\n"
    "args:\n"
    "  - '--debug'\n"
    "env:\n"
    "  PYTHONPATH: .",
)
_MAP_EXAMPLES = (
    "text '['apple', 'banana', 'kiwi', 'cat']' all fruits should become dogs =>['dog', 'dog', 'dog', "
    "'cat']",
    "text 'this is a string' convert vowels to numbers =>'th1s 1s 4 str1ng'",
    "text '('small', 'tiny', 'huge', 'enormous')' convert size adjectives to numbers 1-10 =>'(2, 1, "
    "8, 10)'",
    "text '{'happy', 'sad', 'angry', 'joyful'}' convert emotions to colors =>'{'yellow', 'blue', "
    "'red', 'gold'}'",
    "text '{'item1': 'apple', 'item2': 'banana', 'item3': 'cat'}' convert fruits to vegetables "
    "=>'{'item1': 'carrot', 'item2': 'broccoli', 'item3': 'cat'}'",
    "text '[10, 20, 30, 40]' double each number =>'[20, 40, 60, 80]'",
    "text 'HELLO' make consonants lowercase =>'hEllO'",
)
_FORMAT_EXAMPLES = (
    "text 1 format 'number to text' =>one",
    "text 'apple' format 'company' =>Apple Inc.",
    "text 'fetch logs\n"
    "| fields timestamp, severity\n"
    "| fieldsAdd severity = lower(loglevel)' format 'Japanese' =>fetch ログ\n"
    "| fields タイムスタンプ、重大度\n"
    "| fieldsAdd 重大度 = lower(ログレベル)",
    "text 'Hi mate, how are you?' format 'emoji' =>Hi mate, how are you? 😊",
    "text 'Hi mate, how are you?' format 'Italian' =>Ciao amico, come stai?",
    "text 'Sorry, everyone. But I will not be able to join today.' format 'japanese' "
    "=>すみません、皆さん。でも、今日は参加できません。",
    "text 'Sorry, everyone. But I will not be able to join today.' format 'japanese romanji' "
    "=>Sumimasen, minasan. Demo, kyō wa sanka dekimasen.",
    "text 'April 1, 2020' format 'EU date' =>01.04.2020",
    "text '23' format 'binary' =>10111",
    "text '77' format 'hexadecimal' =>0x4D",
    "text '{\n"
    '    "name": "Manual Game",\n'
    '    "type": "python",\n'
    '    "request": "launch",\n'
    '    "program": "${workspaceFolder}/envs/textgrid.py",\n'
    '    "cwd": "${workspaceFolder}",\n'
    '    "args": [\n'
    '        "--debug"\n'
    "    ],\n"
    '    "env": {\n'
    '        "PYTHONPATH": "."\n'
    "    }\n"
    "}' format 'yaml' =>name: Manual Game\n"
    "type: python\n"
    "request: launch\n"
    "program: ${workspaceFolder}/envs/textgrid.py\n"
    "cwd: ${workspaceFolder}\n"
    "args:\n"
    "  - '--debug'\n"
    "env:\n"
    "  PYTHONPATH: .",
)
_REPLACE_EXAMPLES = (
    "text 'a + b' replace 'b' with '' =>a",
    "text 'a + b' replace 'c' with '' =>a + b",
    "text 'SELECT title, author, pub_date FROM catalog WHERE pub_date = 2021;' replace 'WHERE ...' "
    "with '' =>SELECT title, author, pub_date FROM catalog;",
    "text 'a + b ^ 2' replace 'b' with '' =>a",
    "text '(a + b)^2 - 6 = 18' replace 'b' with '' =>a^2 - 6 = 18",
    "text 'The green fox jumps of the brown chair.' replace 'green' with 'red' =>The red fox jumps of "
    "the brown chair.",
    "text 'My telephone number is +43 660 / 453 4436 88.' replace '6' with '4' =>My telephone number "
    "is +43 440 / 453 4434 88.",
    "text 'I like to eat apples, bananas and oranges.' replace 'fruits' with 'vegetables' =>I like to "
    "eat tomatoes, carrots and potatoes.",
    "text 'Our offices are in London, New York and Tokyo.' replace 'London | New York | Tokyo' with "
    "'Madrid | Vienna | Bucharest' =>Our offices are in Madrid, Vienna and Bucharest.",
    "text 'The number Pi is 3.14159265359' replace '3.1415926...' with '3.14' =>The number Pi is "
    "3.14.",
    "text 'She likes all books about Harry Potter.' replace 'harry potter' with 'Lord of the Rings' "
    "=>She likes all books about Lord of the Rings.",
    "text 'What is the capital of the US?' replace 'Test' with 'Hello' =>What is the capital of the "
    "US?",
    "text 'Include the following files: file1.txt, file2.txt, file3.txt' replace '*.txt' with "
    "'*.json' =>Include the following files: file1.json, file2.json, file3.json",
    "text 'I like 13 Samurai, Pokemon and Digimon' replace 'Pokemon' with '' =>I like 13 Samurai and "
    "Digimon",
    "text 'This product is fucking stupid. The battery is weak. Also, the delivery guy is a moran, "
    "and probably scratched the cover.' replace 'hate speech comments' with '' =>The battery of the "
    "product is weak. Also, the delivery guy probably scratched the cover.",
)
_INCLUDE_EXAMPLES = (
    "text 'The green fox jumps of the brown chair.' include 'in the living room' =>In the living room "
    "the red fox jumps of the brown chair.",
    "text 'Anyone up for Argentina vs Croatia tonight?.' include 'place: Linz' =>Anyone up for "
    "Argentina vs Croatia in Linz tonight?",
    "text 'We received a model BL-03W as a gift and have been impressed by the power it has to pick "
    "up dirt, pet hair, dust on hard surfaces.' include 'details about the black color of the model "
    "and the low price' =>We received a black model BL-03W as a gift and have been impressed by the "
    "power it has to pick up dirt, pet hair, dust on hard surfaces. The low price is also a plus.",
    "text 'I like to eat apples, bananas and oranges.' include 'mangos, grapes, passion fruit' =>I "
    "like to eat apples, bananas, oranges, mangos, grapes and passion fruit.",
    "text 'Our offices are in London, New York and Tokyo.' include 'Madrid, Vienna, Bucharest' =>Our "
    "offices are in London, New York, Tokyo, Madrid, Vienna and Bucharest.",
    "text 'Tonight, on the 20th of July, we will have a party in the garden.' include 'at 8pm' "
    "=>Tonight at 8pm, on the 20th of July, we will have a party in the garden.",
    "text '[1, 2, 3, 4]' include '5' =>[1, 2, 3, 4, 5]",
    "text '[1, 2, 3, 4]' include 'prepend 5' =>[5, 1, 2, 3, 4]",
    "text 'fetch logs | fieldsAdd severity = lower(loglevel)' include '| fields `severity` next to "
    "fetch |' =>fetch logs | fields severity | fieldsAdd severity = lower(loglevel)",
)
_COMBINE_EXAMPLES = (
    "1 + 2 =>3",
    "'x' + 1 =>x + 1",
    "y + 2 =>y + 2",
    "'1' + 2 =>3",
    "17 + 'pi' =>20.1415926535...",
    "7.2 + 'five' =>12.2",
    "True + 0 => False",
    "False + 'True' =>False",
    "['a', 'b'] + ['c', 'd'] =>['a', 'b', 'c', 'd']",
    "False + 1 =>False",
    "True + True =>True",
    "False + False =>False",
    "'apple' + 'banana' =>apple, banana",
    "['apple'] + 'banana' =>['apple', 'banana']",
    "'Hi, I am Alan. I am 23 years old.' + 'I like to play football.' =>Hi, I am Alan. I am 23 years "
    "old. I like to play football.",
    "'We have five red cars' + 'and two blue ones.' =>We have five red cars and two blue ones.",
    "'Zero' + 1 =>1",
    "'One' + 'Two' =>3",
    "'Three' + 4 =>7",
    "'a + b' + 'c + d' =>a + b + c + d",
    "'My cat has four legs equals to x. If x1 (front leg) goes with a velocity of ...' + 'y = 3x + 2' "
    "=>My cat has four legs equals to x. If x1 (front leg) goes with a velocity of ... y = 3x + 2",
    "'x1, x2, x3' + 'y1, y2, y3' =>x1, x2, x3, y1, y2, y3",
    "'house | car | boat' + 'plane | train | ship' =>house | car | boat | plane | train | ship",
    "'The green fox jumps of the brown chair.' + 'The red fox jumps of the brown chair.' =>A green "
    "and a red fox jump of the brown chair.",
)
_EXTRACT_EXAMPLES = (
    "from 'My name is Ashly Johnson. Nice to meet you!' extract 'Full Name' =>Ashly Johnson",
    "from '['Action: a Value: 0.9', 'Action: b Value 0.9', 'Action: c Value: 0.4', 'Action: d Value: "
    "0.0']' extract 'list of letters where Action: * Value: 0.9' =>a | b",
    "from '['Action: d Value: 0.90', 'Action: l Value: 0.62', 'Action: r Value: -inf', 'Action: u "
    "Value: 0.62']' extract 'list of letters where Action: * Value: 0.9' =>d",
    "from '['Action: d Value: 0.76', 'Action: l Value: 1.0', 'Action: r Value: -inf', 'Action: u "
    "Value: 0.62']' extract 'list of highest Value: *' =>1.0",
    "from '['Action: d Value: 0.90', 'Action: l Value: 0.90', 'Action: r Value: -inf', 'Action: u "
    "Value: 0.62']' extract 'list of letters where Action: * Value: smallest' =>r",
    "from 'This is my private number +43 660 / 453 4438 88. And here is my office number +43 (0) 750 "
    "/ 887 387 32-3 Call me when you have time.' extract 'Phone Numbers' =>+43 660 / 453 4438 88 | "
    "+43 (0) 750 / 887 387 32-3",
    "from 'Visit us on www.example.com to see our great products!' extract 'URL' =>www.example.com",
    "from 'A list of urls: http://www.orf.at, https://www.apple.com, https://amazon.de, "
    "https://www.GOOGLE.com, https://server283.org' extract 'Regex https:\\/\\/([w])*.[a-z]*.[a-z]*' "
    "=>https://www.apple.com | https://amazon.de | https://www.GOOGLE.com",
    "from 'Our company was founded on 1st of October, 2010. We are the largest retailer in the "
    "England.' extract 'Date' =>1st of October, 2010",
    "from 'We count four animals. A cat, two monkeys and a horse.' extract 'Animals and counts' =>Cat "
    "1 | Monkey 2 | Horse 1",
    "from '081109 204525 512 INFO dfs.DataNode$PacketResponder: PacketResponder 2 for block "
    "blk_572492839287299681 terminating' extract 'Regex blk_[{0-9}]*' =>blk_572492839287299681",
    "from '081109 203807 222 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block "
    "blk_-6952295868487656571 terminating' extract 'Regex blk_[{0-9}]' =>081109 | 203807 | 222 | 0 | "
    "6952295868487656571",
    "from 'Follow us on Facebook.' extract 'Company Name' =>Facebook",
    "from 'Joe Biden was born November 20, 1942. Divide the year of the birth date by 26.' extract "
    "'mathematical formula' =>1942 / 26",
    "from 'Help us by providing feedback at our service desk.' extract 'Email' =>None",
    "from 'Call us if you need anything.' extract 'Phone Number' =>None",
    "from 'Exception: Failed to query GPT-3 after 3 retries. Errors: "
    "[InvalidRequestError(message=\"This model's maximum context length is 4097 tokens, however you "
    "requested 5684 tokens (3101 in your prompt; ...' extract 'requested tokens' =>5684",
)


def summarize[T](
    model: LanguageModel,
    source: Symbol[T],
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    function = Function("Summarize the content of the following text:\n")
    return _execute_language(model, function, (f"Text: {value!s}\n",), decode_text)


def translate[T](
    model: LanguageModel,
    source: Symbol[T],
    language: str,
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    _require_text(language, "language")
    function = Function(
        f"Your task is to translate and **only** translate the text into {language}:\n"
    )
    return _execute_language(model, function, (str(value),), decode_text)


def modify[T](
    model: LanguageModel,
    source: Symbol[T],
    changes: str,
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    _require_text(changes, "changes")
    function = Function(
        "Modify the text to match the criteria:\n",
        examples=_MODIFY_EXAMPLES,
    )
    return _execute_language(
        model, function, (f"text '{value!s}' modify '{changes}' =>",), decode_text
    )


def filter[T](
    model: LanguageModel,
    source: Symbol[T],
    criteria: str,
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    _require_text(criteria, "criteria")
    function = Function(
        "Filter the text to retain only information matching the criteria. "
        "Leave matching sentences unchanged:\n"
    )
    return _execute_language(
        model, function, (f"text '{value!s}' criteria '{criteria}' =>",), decode_text
    )


def map[T](
    model: LanguageModel,
    source: Symbol[T],
    instruction: str,
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    _require_text(instruction, "instruction")
    function = Function(
        "Transform each element in the input based on the instruction. "
        "Preserve container type and elements that don't match the instruction:\n",
        examples=_MAP_EXAMPLES,
    )
    return _execute_language(model, function, (f"text '{value!s}' {instruction} =>",), decode_text)


def convert[T](
    model: LanguageModel,
    source: Symbol[T],
    format: str,
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    _require_text(format, "format")
    function = Function(
        f"Translate the following text into {format} format.\n",
        examples=_FORMAT_EXAMPLES,
    )
    return _execute_language(
        model, function, (f"text {value!s} format '{format}' =>",), decode_text
    )


def style[T](
    model: LanguageModel,
    source: Symbol[T],
    description: str,
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    _require_text(description, "description")
    function = Function(
        "Style the data based on best practices and the requested description. "
        "Do not remove or invent content.\n"
    )
    return _execute_language(
        model, function, (f"[FORMAT]: {description}\n[DATA]:\n{value!s}\n",), decode_text
    )


def template[T](
    source: Symbol[T],
    template: str,
    *,
    placeholder: str = "{{placeholder}}",
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    _require_text(template, "template")
    _require_text(placeholder, "placeholder")
    if not placeholder:
        msg = "placeholder must not be empty"
        raise ValueError(msg)

    return Symbol(template.replace(placeholder, str(value)))


def replace[T](
    model: LanguageModel,
    source: Symbol[T],
    old: str,
    new: str,
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    _require_text(old, "old")
    _require_text(new, "new")
    function = Function(
        "Replace text parts by string pattern.\n",
        examples=_REPLACE_EXAMPLES,
    )
    return _execute_language(
        model, function, (f"text '{value!s}' replace '{old}' with '{new}' =>",), decode_text
    )


def include[T](
    model: LanguageModel,
    source: Symbol[T],
    information: str,
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    _require_text(information, "information")
    function = Function(
        "Include information based on description.\n",
        examples=_INCLUDE_EXAMPLES,
    )
    return _execute_language(
        model, function, (f"text '{value!s}' include '{information}' =>",), decode_text
    )


def combine[LeftT, RightT](
    model: LanguageModel,
    left: Symbol[LeftT],
    right: Symbol[RightT],
) -> Symbol[str]:
    left_value = _symbol_value(left, "left")
    right_value = _symbol_value(right, "right")
    function = Function(
        "Add the two data types in a logical way:\n",
        examples=_COMBINE_EXAMPLES,
    )
    return _execute_language(
        model, function, (f"{left_value!s} + {right_value!s} =>",), decode_text
    )


def extract[T](
    model: LanguageModel,
    source: Symbol[T],
    pattern: str,
) -> Symbol[str]:
    value = _symbol_value(source, "source")
    _require_text(pattern, "pattern")
    function = Function(
        "Extract a pattern from text:\n",
        examples=_EXTRACT_EXAMPLES,
    )
    return _execute_language(
        model, function, (f"from '{value!s}' extract '{pattern}' =>",), decode_text
    )
