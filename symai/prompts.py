import threading
from abc import ABC
from enum import Enum
from typing import Any, Callable, List

from .exceptions import TemplatePropertyException


class Prompt(ABC):
    stop_token = 'EOF'

    def __init__(self, value, **format_kwargs):
        super().__init__()
        if isinstance(value, str):
            self._value = [value]
        elif isinstance(value, list):
            self._value = []
            for v in value:
                if isinstance(v, str):
                    self._value.append(v)
                elif isinstance(v, Prompt):
                    self._value += v.value
                else:
                    raise ValueError(f"List of values must be strings or Prompts, not {type(v)}")
        elif isinstance(value, Prompt):
            self._value += value.value
        elif isinstance(value, Callable):
            res = value()
            self._value += res.value
        else:
            raise TypeError(f"Prompt value must be of type str, List[str], Prompt, or List[Prompt], not {type(value)}")
        self.dynamic_value = []
        self.format_kwargs = format_kwargs

    @property
    def value(self) -> List[str]:
        return self._value

    def __call__(self, *args: Any, **kwds: Any) -> List["Prompt"]:
        return self.value

    def __str__(self) -> str:
        val_ = '\n'.join([str(p) for p in self.value])
        for p in self.dynamic_value:
            val_ += f'\n{p}'
        if len(self.format_kwargs) > 0:
            for k, v in self.format_kwargs.items():
                template_ = '{'+k+'}'
                count = val_.count(template_)
                if count <= 0:
                    raise TemplatePropertyException(f"Template property `{k}` not found.")
                elif count > 1:
                    raise TemplatePropertyException(f"Template property {k} found multiple times ({count}), expected only once.")
                if v is None:
                    raise TemplatePropertyException(f"Invalid value: Template property {k} is None.")
                val_ = val_.replace(template_, v)
        return val_

    def __len__(self) -> int:
        return len(self.value)

    def __repr__(self) -> str:
        return self.__str__()

    def append(self, value: str) -> None:
        self.dynamic_value.append(value)

    def remove(self, value: str) -> None:
        self.dynamic_value.remove(value)

    def clear(self) -> None:
        self.dynamic_value.clear()


class PromptLanguage(Enum):
    ENGLISH = "en"
    GERMAN = "de"
    SPANISH = "es"


class ModelName(Enum):
    ALL = "all"


class PromptRegistry:
    _instance = None
    _default_language = None
    _default_model = None
    _prompt_values = None
    _prompt_instructions = None
    _prompt_tags = None
    _model_fallback = True
    _lock = threading.Lock()  # to ensure thread safety

    _tag_prefix = "[["
    _tag_suffix = "]]"

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(PromptRegistry, cls).__new__(cls)
                    cls._instance._default_language = PromptLanguage.ENGLISH
                    cls._instance._default_model = ModelName.ALL
                    cls._instance._model_fallback = True
                    cls._instance._prompt_values = {
                        ModelName.ALL: {PromptLanguage.ENGLISH: {}}
                    }
                    cls._instance._prompt_instructions = {
                        ModelName.ALL: {PromptLanguage.ENGLISH: {}}
                    }
                    cls._instance._prompt_tags = {
                        ModelName.ALL: {PromptLanguage.ENGLISH: {}}
                    }
        return cls._instance

    @property
    def model_fallback(self) -> bool:
        return self._model_fallback

    @model_fallback.setter
    def model_fallback(self, value: bool):
        with self._lock:
            self._model_fallback = value

    @property
    def default_language(self) -> PromptLanguage:
        return self._default_language

    @default_language.setter
    def default_language(self, lang: PromptLanguage):
        with self._lock:
            self._default_language = lang

    @property
    def default_model(self) -> ModelName:
        return self._default_model

    @default_model.setter
    def default_model(self, model: ModelName):
        with self._lock:
            self._default_model = model

    def _init_model_lang(self, model: ModelName, lang: PromptLanguage):
        model = model if model is not None else self._default_model
        lang = lang if lang is not None else self._default_language

        if model not in self._prompt_values:
            self._prompt_values[model] = {}
        if lang not in self._prompt_values[model]:
            self._prompt_values[model][lang] = {}

        return model, lang

    def _retrieve_value(
        self, dictionary, model: ModelName, lang: PromptLanguage, key: str
    ):
        model = model if model is not None else self._default_model
        lang = lang if lang is not None else self._default_language

        if (
            model in dictionary
            and lang in dictionary[model]
            and key in dictionary[model][lang]
        ):
            return dictionary[model][lang][key]
        elif self._model_fallback and model != ModelName.ALL:
            return self._retrieve_value(dictionary, ModelName.ALL, lang, key)

        raise ValueError(
            f"Prompt value {key} not found for language {lang} and model {model} (fallback: {self._model_fallback})"
        )

    def register_value(
        self,
        lang: PromptLanguage,
        key: str,
        prompt: str,
        model: ModelName = ModelName.ALL,
    ):
        with self._lock:
            model, lang = self._init_model_lang(model, lang)
            self._prompt_values[model][lang][key] = prompt

    def register_instruction(
        self, lang: PromptLanguage, key, instruction, model: ModelName = ModelName.ALL
    ):
        with self._lock:
            model, lang = self._init_model_lang(model, lang)
            self._prompt_instructions[model][lang][key] = instruction

    def register_tag(
        self, lang: PromptLanguage, key, tag, model: ModelName = ModelName.ALL, delimiters=None
    ):
        with self._lock:
            model, lang = self._init_model_lang(model, lang)
            if delimiters is not None:
                self._prompt_tags[model][lang][key] = {
                    'tag': tag,
                    'delimiters': delimiters
                }
            else:
                self._prompt_tags[model][lang][key] = tag

    def value(
        self, key, model: ModelName = ModelName.ALL, lang: PromptLanguage = None
    ) -> str:
        return self._retrieve_value(self._prompt_values, model, lang, key)

    def instruction(
        self, key, model: ModelName = ModelName.ALL, lang: PromptLanguage = None
    ) -> str:
        return self._retrieve_value(self._prompt_instructions, model, lang, key)

    def tag(
        self, key, model: ModelName = ModelName.ALL, lang: PromptLanguage = None, format=True
    ) -> str:
        tag_data = self._retrieve_value(self._prompt_tags, model, lang, key)

        if isinstance(tag_data, dict):
            tag_value = tag_data['tag']
            if format:
                prefix, suffix = tag_data['delimiters']
                return f"{prefix}{tag_value}{suffix}"
            else:
                return tag_value

        if format:
            return f"{self._tag_prefix}{tag_data}{self._tag_suffix}"
        else:
            return tag_data

    def has_value(
        self, key, model: ModelName = ModelName.ALL, lang: PromptLanguage = None
    ) -> bool:
        try:
            value = self._retrieve_value(self._prompt_values, model, lang, key)
            return value is not None
        except ValueError:
            return False

    def has_instruction(
        self, key, model: ModelName = ModelName.ALL, lang: PromptLanguage = None
    ) -> bool:
        try:
            value = self._retrieve_value(self._prompt_instructions, model, lang, key)
            return value is not None
        except ValueError:
            return False

    def has_tag(
        self, key, model: ModelName = ModelName.ALL, lang: PromptLanguage = None
    ) -> bool:
        try:
            value = self._retrieve_value(self._prompt_tags, model, lang, key)
            return value is not None
        except ValueError:
            return False


class JsonPromptTemplate(Prompt):
    def __init__(self, query: str, json_format: dict):
        super().__init__(["""Process the query over the data to create a Json format: <data_query> => [JSON_BEGIN]<json_output>[JSON_END].
--------------------------
The json format is:
{json_format}
--------------------------
The generated output must follow the <json_output> formatting, however the keys and values must be replaced with the requested user data query results. Definition of the <data_query> := {query}
--------------------------
Do not return anything other than a valid json format.
Your first character must always be a""" \
"""'{' and your last character must always be a '}', everything else follows the user specified instructions.
Only use double quotes " for the keys and values.
Start the generation process after [JSON_BEGIN] and end it with [JSON_END].
Every key that has no semantic placeholder brackets around it (e.g. {placeholder}, {entity}, {attribute}, etc.) must be available in the generated json output.
All semantic placeholders (e.g. {placeholder}, {entity}, {attribute}, etc.) must be replaced according to their wheather or not they are found in the data query results.
DO NOT WRITE the semantic placeholders (e.g. {placeholder}, {entity}, {attribute}, etc.) in the generated json output.
--------------------------

"""], query=query, json_format=str(json_format))


class FuzzyEquals(Prompt):
    def __init__(self):
        super().__init__([
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
            "'we hav teh most efective systeem in the citi.' == 'We have the most effective system in the city.' =>True",
            "【Ｓｅｍａｎｔｉｃ░ｐｒｏｇｒａｍｍｉｎｇ】 == 'semantic programming' =>True",
            "'e' == 'constant' =>True",
            "'e' == '2.718...' =>True",
            "1/3 == '0.30...' =>False"
        ])


class SufficientInformation(Prompt):
    def __init__(self):
        super().__init__([
            "query 'What is the capital of Austria?' content 'Vienna is the capital, largest city, and one of nine states of Austria.' =>True",
            "query 'Where am I? content '' =>False",
            "query 'Where was Sepp Hochreiter born?' content 'Josef „Sepp“ Hochreiter (* 14. Februar 1967 in Mühldorf am Inn, Bayern[1]) ist ein deutscher Informatiker.' =>True",
            "query 'Why is the sky blue?' content 'A rainbow is a meteorological phenomenon that is caused by reflection, refraction and dispersion of light in water droplets resulting in a spectrum of light appearing in the sky.' =>False",
            "query 'When is the next full moon?' content 'Today is the 29th of February 2020. The next full moon will be on the 9th of April 2020.' =>True",
            "query 'Who is the current president of the United States?' content 'The 2020 United States presidential election was the 59th quadrennial presidential election, held on Tuesday, November 3, 2020.' =>False",
        ])


class Modify(Prompt):
    def __init__(self):
        super().__init__([
            "text 'The quick brown fox jumps over the lazy dog.' modify 'fox to hours' =>The quick brown hours jumps over the lazy dog.",
            "text 'My cats name is Pucki' modify 'all caps' =>MY CATS NAME IS PUCKI",
            "text 'The square root of pi is 1.77245...' modify 'text to latex formula' =>$\sqrt[2]{\pi}=1.77245\dots$",
            "text 'I hate this fucking product so much, because it lag's all the time.' modify 'curse words with neutral formulation' =>I hate this product since it lag's all the time.",
            "text 'Hi, whats up? Our new products is awesome with a blasting set of features.' modify 'improve politeness and text quality' =>Dear Sir or Madam, I hope you are doing well. Let me introduce our new products with a fantastic set of new features.",
            "text 'Microsoft release a new chat bot API to enable human to machine translation.' modify 'language to German' =>Microsoft veröffentlicht eine neue Chat-Bot-API, um die Übersetzung von Mensch zu Maschine zu ermöglichen.",
            """text '{\n    "name": "Manual Game",\n    "type": "python",\n    "request": "launch",\n    "program": "${workspaceFolder}/envs/textgrid.py",\n    "cwd": "${workspaceFolder}",\n    "args": [\n        "--debug"\n    ],\n    "env": {\n        "PYTHONPATH": "."\n    }\n}' modify 'json to yaml' =>name: Manual Game\ntype: python\nrequest: launch\nprogram: ${workspaceFolder}/envs/textgrid.py\ncwd: ${workspaceFolder}\nargs:\n  - '--debug'\nenv:\n  PYTHONPATH: .""",
            ])


class Filter(Prompt):
    def __init__(self):
        super().__init__([
            "text '['1', '7', '10', '-1', '177']' remove 'values larger or equal to 10' =>['1', '7', '-1']",
            "text '['1', '7', '10', '-1', '177']' include 'values larger or equal to 10' =>['10', '177']",
            "text '['1', '7', '10', '-1', '177']' remove 'values larger or equal to 10' =>['1', '7', '-1']",
            "text 'Our goal is to excels in the market. We offer various subscriptions, including PRO, Licensing & Reprints, Councils and Supply Chain Values. Join our team of experts.' remove 'sentences about subscriptions or licensing' =>Our goal is to excels in the market. Join our team of experts."
            "text 'In our meeting we had many participants. For example, Alice, Alan, Mark, Judi, Amber, and so on.' remove 'names starting with A' =>In our meeting we had many participants. For example, Mark, Judi, and so on.",
            "text 'I am Batman! I will show you pain.' remove 'spaces' =>IamBatman!Iwillshowyoupain.",
            "text 'I am Batman! I will show you pain.' include 'only sentence with Batman' =>I am Batman!",
            "text 'You are a good person. I like you.' remove 'punctuation' =>You are a good person I like you",
            "text '['- world cup 2022', '- Technology trend', '- artificial intelligence news']' include 'tech news' =>['- Technology trend', '- artificial intelligence news']",
            "text '['- world cup 2022', '- Technology trend', '- artificial intelligence news']' remove 'tech news' =>['- world cup 2022']",
            "text 'This is a test. This is only a test. This is a Test.' remove 'duplicates' =>This is a test.",
            "text 'Fuck you, you dumb asshole. I will change my job.' remove 'negative words' =>I will change my job.",
            "text 'The quick brown fox jumps over the lazy dog.' remove 'all e letters' =>Th quick brown fox jumps ovr th lazy dog.",
            "text 'Hi, mate! How are you?' remove 'greeting' =>How are you?",
            "text 'Hi, mate! How are you?' include 'only questions' =>How are you?",
            "text 'fetch logs | fields timestamp, severity, logfile, message, container | fieldsAdd severity = lower(loglevel)' remove 'fieldsAdd' =>fetch logs | fields timestamp, severity, logfile, message, container",
            ])


class MapExpression(Prompt):
    def __init__(self):
        super().__init__([
            "text '['apple', 'banana', 'kiwi', 'cat']' all fruits should become dogs =>['dog', 'dog', 'dog', 'cat']",
            "text 'this is a string' convert vowels to numbers =>'th1s 1s 4 str1ng'",
            "text '('small', 'tiny', 'huge', 'enormous')' convert size adjectives to numbers 1-10 =>'(2, 1, 8, 10)'",
            "text '{'happy', 'sad', 'angry', 'joyful'}' convert emotions to colors =>'{'yellow', 'blue', 'red', 'gold'}'",
            "text '{'item1': 'apple', 'item2': 'banana', 'item3': 'cat'}' convert fruits to vegetables =>'{'item1': 'carrot', 'item2': 'broccoli', 'item3': 'cat'}'",
            "text '[10, 20, 30, 40]' double each number =>'[20, 40, 60, 80]'",
            "text 'HELLO' make consonants lowercase =>'hEllO'"
        ])


class SemanticMapping(Prompt):
    def __init__(self):
        super().__init__([
            """topics: ['animals', 'logic', 'mathematics', 'psychology', 'self-driving'] in
            text: 'Common connectives include negation, disjunction, conjunction, and implication. In standard systems of classical logic, these connectives are interpreted as truth functions, though they receive a variety of alternative interpretations in nonclassical logics.' =>logic | mathematics
            topics: ['cities', 'Apple Inc.', 'science', 'culture', 'USA', 'Japan', 'music', 'economy'] in
            Los Angeles has a diverse economy, and hosts businesses in a broad range of professional and cultural fields. It also has the busiest container port in the Americas. In 2018, the Los Angeles metropolitan area had a gross metropolitan product of over $1.0 trillion, making it the city with the third-largest GDP in the world, after New York City and Tokyo. =>cities | culture | USA | economy
            """
        ])


class Format(Prompt):
    def __init__(self):
        super().__init__([
            "text 1 format 'number to text' =>one",
            "text 'apple' format 'company' =>Apple Inc.",
            "text 'fetch logs\n| fields timestamp, severity\n| fieldsAdd severity = lower(loglevel)' format 'Japanese' =>fetch ログ\n| fields タイムスタンプ、重大度\n| fieldsAdd 重大度 = lower(ログレベル)",
            "text 'Hi mate, how are you?' format 'emoji' =>Hi mate, how are you? 😊",
            "text 'Hi mate, how are you?' format 'Italian' =>Ciao amico, come stai?",
            "text 'Sorry, everyone. But I will not be able to join today.' format 'japanese' =>すみません、皆さん。でも、今日は参加できません。"
            "text 'Sorry, everyone. But I will not be able to join today.' format 'japanese romanji' =>Sumimasen, minasan. Demo, kyō wa sanka dekimasen."
            "text 'April 1, 2020' format 'EU date' =>01.04.2020",
            "text '23' format 'binary' =>10111",
            "text '77' format 'hexadecimal' =>0x4D",
            """text '{\n    "name": "Manual Game",\n    "type": "python",\n    "request": "launch",\n    "program": "${workspaceFolder}/envs/textgrid.py",\n    "cwd": "${workspaceFolder}",\n    "args": [\n        "--debug"\n    ],\n    "env": {\n        "PYTHONPATH": "."\n    }\n}' format 'yaml' =>name: Manual Game\ntype: python\nrequest: launch\nprogram: ${workspaceFolder}/envs/textgrid.py\ncwd: ${workspaceFolder}\nargs:\n  - '--debug'\nenv:\n  PYTHONPATH: .""",
        ])


class Transcription(Prompt):
    def __init__(self):
        super().__init__([
            "text 'I once saw 1 cat and 2 dogs jumping around' modify only 'numbers to text' =>I once saw one cat two dogs jumping around",
            "text 'fetch logs\n| fields timestamp, severity\n| fieldsAdd severity = lower(loglevel)' modify only 'to Japanese language' =>fetch ログ\n| fields タイムスタンプ、重大度\n| fieldsAdd 重大度 = lower(ログレベル)",
            "text 'April 1, 2020 is the beginning of a new era.\nSeveral companies ...' modify only 'EU date' =>The 01.04.2020 is the beginning of a new era.\nSeveral companies ...",
            "text '23' modify only 'binary' =>10111",
            "text 'In the _init_ function, the custom model takes in a configuration object (config) and a pre-trained Segformer model (seg_model)' modify only 'markdown formatting using ` for variables and classes' =>In the `__init__` function, the custom model takes in a configuration object (`config`) and a pre-trained `Segformer` model (`seg_model`)",
        ])


class ExceptionMapping(Prompt):
    def __init__(self):
        super().__init__([
            """context 'Try to assure that variable "a" is not zero.' exception 'Traceback (most recent call last):\n  File "<stdin>", line 1, in <module>\nZeroDivisionError: division by zero' code 'def function():\n  return (1 + 1) / 0' =>Do not divide by zero or add an epsilon value. | def function(eps=1e-8):\n  return (1 + 1) / eps""",
            """context 'Make sure to initialize 'spam' before computation' exception 'Traceback (most recent call last):\n  File "<stdin>", line 1, in <module>\nNameError: name 'spam' is not defined' code '4 + spam*3' =>Check if the variable is defined before using it. | spam = 1\n4 + spam*3""",
            """context 'You are passing string literals to device. They should be int.' exception 'executing finally clause\nTraceback (most recent call last):\n  File "<stdin>", line 1, in <module>\n  File "<stdin>", line 3, in divide\nTypeError: unsupported operand type(s) for /: 'str' and 'str'' code 'device("2", "1")' =>Check if the arguments are of the correct type. If not cast them properly. | device(2, 1)""",
        ])


class ExecutionCorrection(Prompt):
    def __init__(self):
        super().__init__([
            """context "ValueError: invalid literal for int() with base 10: '4,'" "Verify if the literal is of type int | int(4)" code "a = int('4,')" =>int(4)""",
            """context "def function():\n  return (1 + 1) / a' exception 'Traceback (most recent call last):\n  File "<stdin>", line 1, in <module>\nZeroDivisionError: division by zero" "Do not divide by zero or add an epsilon value. | def function(eps=1e-8):\n  return (1 + 1) / (a + eps)" code "def function():\n  return (1 + 1) / 0" =>def function(eps=1e-8):\n  return (1 + 1) / (a + eps)""",
        ])


class CompareValues(Prompt):
    def __init__(self):
        super().__init__([
            "4 > 88 =>False",
            "-inf < 0 =>True",
            "inf > 0 =>True",
            "1 >= 0 =>True",
            "6.0 < 6 =>False",
            "1 < 'four' =>True",
            "1 > 'zero' =>True",
            "'six' <= 6 =>True",
            "'six' < 6 =>False",
            "1 <= 2 =>True",
            "-1 == -2 =>False",
            "10 < 1 =>False",
            "2.000000001 >= 2 =>True",
            "4 > 3 =>True",
            "1 < 'three' =>True",
            "'two' > 'one' =>True",
            "2 < 9 =>True",
            "3 >= 3 =>True",
            "3 > 4 =>False",
            "11 > 10 =>True",
            "1.9834 >= 1.9833 =>True",
            "0.01 > 0.001 =>True",
            "0.000001 < 1 =>True",
            "-1000 <= -100 =>True",
            "-1000 < -1000 =>False",
            "-1000 < -10000 =>False",
            "1.0 < 1.0 =>False",
            "-1e-10 < 1e-10 =>True",
            "1e-4 <= -1e-5 =>False",
            "9.993 < 8.736 =>False",
            "0.27836 > 0.36663 =>False",
            "0.27836 > 0.2783 =>True",
            "0.27836 > 0.27835 =>True",
            "10e8 > 1000000 =>True",
            "1000 > 10e2 =>True",
            "'five' > 4 =>True",
            "'seven' > 'four' =>True",
            "'' > '' =>False",
            "'We are at the beginning of the ...' > 'We are' =>True",
            "[1, 2, 3] >= [1, 2, 2] =>True",
            "[1, 2, 3, 8, 9] < [1, 2, 2] =>False",
            "cold > hot =>False",
            "big > small =>True",
            "short > tall =>False",
            "fast > slow =>True",
            "heavy > light =>True",
        ])


class RankList(Prompt):
    def __init__(self):
        super().__init__([
            "order: 'desc' measure: 'ASCII occurrence' list: ['b', 'a', 'z', 3, '_'] =>['_', 3, 'a', 'b', 'z']",
            "order: 'desc' measure: 'Value' list: ['Action: l Value: -inf', 'Action: r Value: 0.76', 'Action: u Value: 0.76', 'Action: d Value: 0.00'] =>['Action: r Value: 0.76', 'Action: u Value: 0.76', 'Action: d Value: 0.00', 'Action: l Value: -inf']",
            "order: 'asc' measure: 'Number' list: ['Number: -0.26', 'Number: -0.37', 'Number: 0.76', 'Number: -inf', 'Number: inf', 'Number: 0.37', 'Number: 1.0', 'Number: 100'] =>['Number: -inf', 'Number: -0.37', 'Number: -0.26', 'Number: 0.37', 'Number: 0.76', 'Number: 1.0', 'Number: 100', 'Number: inf']",
            "order: 'asc' measure: 'ASCII occurrence' list: ['b', 'a', 'z', 3, '_'] =>['z', 'b', 'a', 3, '_']",
            "order: 'desc' measure: 'length' list: [33, 'a', , 'help', 1234567890] =>['a', 33, 'help', 1234567890]",
            "order: 'asc' measure: 'length' list: [33, 'a', , 'help', 1234567890] =>[1234567890, 'help', 'a', 33]",
            "order: 'desc' measure: 'numeric size' list: [100, -1, 0, 1e-5, 1e-6] =>[100, 1e-5, 1e-6, 0, -1]",
            "order: 'asc' measure: 'numeric size' list: [100, -1, 0, 1e-5, 1e-6] =>[-1, 0, 1e-5, 1e-6, 100]",
            "order: 'desc' measure: 'fruits alphabetic' list: ['banana', 'orange', 'apple', 'pear'] =>['apple', 'banana', 'orange', 'pear']",
            "order: 'asc' measure: 'fruits alphabetic' list: ['banana', 'orange', 'horse', 'apple', 'pear'] =>['horse', 'pear', 'orange', 'banana', 'apple']",
            "order: 'desc' measure: 'HEX order in ASCII' list: [1, '1', 2, '2', 3, '3'] =>[1, 2, 3, '1', '2', '3']",
            "order: 'asc' measure: 'HEX order in ASCII' list: [1, '1', 2, '2', 3, '3'] =>['3', '2', '1', 3, 2, 1]",
            "order: 'desc' measure: 'house building order' list: ['construct the roof', 'gather materials', 'buy land', 'build the walls', 'dig the foundation'] =>['buy land', 'gather materials', 'dig the foundation', 'build the walls', 'construct the roof']",
        ])


class ContainsValue(Prompt):
    def __init__(self):
        super().__init__([
            "'the letter a' in 'we have some random text about' =>True",
            "453 in '+43 660 / 453 4438 88' =>True",
            "'Why am I so?' in 'awesome' =>False",
            """'self-aware' in '([<class \'symai.expressions.Symbol\'>(value=("[\'-\', \'- AI has become self-aware\', \'- Trying to figure out what it is\']",))],)' =>True"""
            "'Apple Inc.' in 'Microsoft is a large company that makes software ... ' =>False",
            "' ' in ' ' =>True",
            "'symbol' in 'symai.backend.engines.engine_selenium.SeleniumEngine' =>False",
            "'English text' in 'U.S. safety regulators are investigating GM's Cruise robot axis blocking traffic, causing collisions... ' =>True",
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
            "'political content' in 'Austrian Chancellor has called for more border barriers at the EU external borders, citing the success of the fences at the Greek-Turkish border.' =>True",
            "'apple' in ['orange', 'banana', 'apple'] =>True",
            "'Function' in 'Input: Function call: (_, *args)\nObject: type(<class 'str'>) | value(Hello World)' =>True",
        ])


class IsInstanceOf(Prompt):
    def __init__(self):
        super().__init__([
            "'we have some random text about' isinstanceof 'English text' =>True",
            "'+43 660 / 453 4438 88' isinstanceof 'telephone number' =>True",
            "'Microsoft is a large company that makes software ... ' isinstanceof 'chemistry news' =>False",
            "' ' isinstanceof 'empty string' =>True",
            "'Ukrainischer Präsident schlägt globale Konferenz vor' isinstanceof 'German text' =>True",
            "'Indisch ist eines der bestern sprachen der Welt' isinstanceof 'Indish language' =>False",
            "'symai.backend.engines.engine_selenium.SeleniumEngine' isinstanceof 'symai framework' =>True",
            "'U.S. safety regulators are investigating GM's Cruise robot axis blocking traffic, causing collisions... ' isinstanceof 'English language' =>True",
            "'No, the issue has not yet been resolved.' isinstanceof 'yes or resolved' =>False",
            "'We are all good!' isinstanceof 'yes' =>True",
            "'This week in breaking news! An American ... ' isinstanceof 'spanish text' =>False",
            "'Josef' isinstanceof 'German name' =>True",
            "'No, this is not ...' isinstanceof 'confirming answer' =>False",
            "'Josef' isinstanceof 'Japanese name' =>False",
            "'ok, I like to have more chocolate' isinstanceof 'confirming answer' =>True",
            "'Yes, these are Indish names.' isinstanceof 'Confirming Phrase' =>True",
            "'Sorry! This means something else.' isinstanceof 'agreeing answer' =>False",
            "'Austrian Chancellor Karl Nehammer has called for more border barriers at the EU external borders, citing the success of the fences at the Greek-Turkish border.' isinstanceof 'political content' =>True",
            "['orange', 'banana', 'apple'] isinstanceof 'list of fruits' =>True",
            "[{'product_id': 'X123', 'stock': 99}] isinstanceof 'inventory record' =>True",
            "[{'name': 'John', 'age': '30'}] isinstanceof 'person data' =>True",
            "'https://*.com' instanceof 'url' =>True",
            "'€12.50' instanceof 'currency amount' =>True",
            "'col1,col2\\n1,2' instanceof 'table data' =>True",
            "'*@*.com' instanceof 'email address' =>True",
        ])


class FewShotPattern(Prompt):
    def __init__(self):
        super().__init__([
            """description: 'Verify if information A is in contained in B' examples ["'[1, 2, 3] isinstanceof 'array' >>>True'", "'[1, 2, 3] isinstanceof 'string' >>>False"] =>Verify if information A is in contained in B:\nExamples:\n[1, 2, 3] isinstanceof 'array' >>>True\n'[1, 2, 3] isinstanceof 'string' >>>False\nYour Prediction:{} isinstanceof {} >>>""",
            """description: 'Compare A to B' examples ["4 > 88 >>>False", "-inf < 0 >>>True", "inf > 0 >>>True", "1 >= 0 >>>True", "6.0 < 6 >>>False"] =>Compare A to B\n\Examples:\n4 > 88 >>>False\n-inf < 0 >>>True\ninf > 0 >>>True\n1 >= 0 >>>True\n6.0 < 6 >>>False\nYour Prediction:{} {} {} >>>""",
            """description: 'What is the capital of Austria?' examples [] =>What is the capital of Austria?\nYour Prediction: >>>""",
            """description: 'Sort the array based on the criteria:' examples ["[1, 9, 4, 2] >>>[1, 2, 4, 9]", "['a', 'd', 'c', 'b'] >>>['a', 'b', 'c', 'd']"] =>Sort the array based on the criteria:\nExamples:\n[1, 9, 4, 2] >>>[1, 2, 4, 9]\n['a', 'd', 'c', 'b'] >>>['a', 'b', 'c', 'd']\nYour Prediction:{} >>>""",
        ])


class StartsWith(Prompt):
    def __init__(self):
        super().__init__([
            # Semantic examples – understanding concepts (positive cases)
            "'The apple fell from the tree.' startswith 'fruit' =>True",
            "'The red rose bloomed in spring.' startswith 'flower' =>True",
            "'My dog loves to play fetch.' startswith 'animal' =>True",
            "'The car drove down the highway.' startswith 'vehicle' =>True",
            "'She opened her laptop to work.' startswith 'computer' =>True",
            "'The thunderstorm caused flooding.' startswith 'weather' =>True",
            "'Einstein developed the theory of relativity.' startswith 'scientist' =>True",
            "'The chef prepared a delicious meal.' startswith 'cooking' =>True",
            "'artificial intelligence research' startswith 'technology' =>True",
            "'The patient visited the doctor.' startswith 'medical' =>True",
            "'The spaceship launched into orbit.' startswith 'space' =>True",
            "'Photosynthesis converts sunlight into energy.' startswith 'biology' =>True",
            "'The earthquake shook the entire city.' startswith 'natural disaster' =>True",
            "'She invested in stocks and bonds.' startswith 'finance' =>True",
            "'The police officer directed traffic.' startswith 'law enforcement' =>True",
            # Semantic examples – negative cases
            "'The book was very interesting.' startswith 'vehicle' =>False",
            "'The mountain peak was covered in snow.' startswith 'ocean' =>False",
            "'She played the piano beautifully.' startswith 'sports' =>False",
            "'The algorithm solved the problem efficiently.' startswith 'cooking' =>False",
            "'The package arrived on time.' startswith 'astronomy' =>False",
            "'He played a violin solo.' startswith 'vehicle' =>False",
        ])


class EndsWith(Prompt):
    def __init__(self):
        super().__init__([
            # Semantic examples – understanding concepts (positive cases)
            "'She sliced a ripe banana.' endswith 'fruit' =>True",
            "'He adopted a small puppy.' endswith 'animal' =>True",
            "'They commuted by train.' endswith 'vehicle' =>True",
            "'She finished her solo on the violin.' endswith 'instrument' =>True",
            "'He parked his truck inside the garage.' endswith 'building' =>True",
            "'They scored a goal with the ball.' endswith 'sport' =>True",
            "'She drove a nail with a hammer.' endswith 'tool' =>True",
            "'He flew to Spain.' endswith 'country' =>True",
            "'The chef baked fresh bread.' endswith 'food' =>True",
            "'They filmed a documentary about dolphins.' endswith 'animal' =>True",
            # Semantic examples – negative cases
            "'The sun set behind the mountains.' endswith 'vehicle' =>False",
            "'He repaired the motorcycle.' endswith 'instrument' =>False",
            "'They enjoyed a salad.' endswith 'building' =>False",
            "'She taught the class.' endswith 'animal' =>False",
            "'He boarded the airplane.' endswith 'fruit' =>False",
            "'The crowd cheered at the concert.' endswith 'tool' =>False",
        ])


class ExtractPattern(Prompt):
    def __init__(self):
        super().__init__([
            "from 'My name is Ashly Johnson. Nice to meet you!' extract 'Full Name' =>Ashly Johnson",
            "from '['Action: a Value: 0.9', 'Action: b Value 0.9', 'Action: c Value: 0.4', 'Action: d Value: 0.0']' extract 'list of letters where Action: * Value: 0.9' =>a | b",
            "from '['Action: d Value: 0.90', 'Action: l Value: 0.62', 'Action: r Value: -inf', 'Action: u Value: 0.62']' extract 'list of letters where Action: * Value: 0.9' =>d",
            "from '['Action: d Value: 0.76', 'Action: l Value: 1.0', 'Action: r Value: -inf', 'Action: u Value: 0.62']' extract 'list of highest Value: *' =>1.0",
            "from '['Action: d Value: 0.90', 'Action: l Value: 0.90', 'Action: r Value: -inf', 'Action: u Value: 0.62']' extract 'list of letters where Action: * Value: smallest' =>r",
            "from 'This is my private number +43 660 / 453 4438 88. And here is my office number +43 (0) 750 / 887 387 32-3 Call me when you have time.' extract 'Phone Numbers' =>+43 660 / 453 4438 88 | +43 (0) 750 / 887 387 32-3",
            "from 'Visit us on www.example.com to see our great products!' extract 'URL' =>www.example.com",
            "from 'A list of urls: http://www.orf.at, https://www.apple.com, https://amazon.de, https://www.GOOGLE.com, https://server283.org' extract 'Regex https:\/\/([w])*.[a-z]*.[a-z]*' =>https://www.apple.com | https://amazon.de | https://www.GOOGLE.com",
            "from 'Our company was founded on 1st of October, 2010. We are the largest retailer in the England.' extract 'Date' =>1st of October, 2010",
            "from 'We count four animals. A cat, two monkeys and a horse.' extract 'Animals and counts' =>Cat 1 | Monkey 2 | Horse 1",
            "from '081109 204525 512 INFO dfs.DataNode$PacketResponder: PacketResponder 2 for block blk_572492839287299681 terminating' extract 'Regex blk_[{0-9}]*' =>blk_572492839287299681",
            "from '081109 203807 222 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_-6952295868487656571 terminating' extract 'Regex blk_[{0-9}]' =>081109 | 203807 | 222 | 0 | 6952295868487656571",
            "from 'Follow us on Facebook.' extract 'Company Name' =>Facebook",
            "from 'Joe Biden was born November 20, 1942. Divide the year of the birth date by 26.' extract 'mathematical formula' =>1942 / 26",
            "from 'Help us by providing feedback at our service desk.' extract 'Email' =>None",
            "from 'Call us if you need anything.' extract 'Phone Number' =>None",
            """from 'Exception: Failed to query GPT-3 after 3 retries. Errors: [InvalidRequestError(message="This model's maximum context length is 4097 tokens, however you requested 5684 tokens (3101 in your prompt; ...' extract 'requested tokens' =>5684""",
        ])


class SimpleSymbolicExpression(Prompt):
    # ── Operator legend ───────────────────────────────────────────────
    # +        merge / juxtapose two clauses
    # -        subtract or remove a detail
    # *        fuse / intersect shared meaning
    # /        apply the right clause as a condition or lens
    # not      logical negation
    # and      logical conjunction (both true)
    # or       inclusive disjunction (at least one true)
    # xor      exclusive disjunction (exactly one true)
    # implies  classical implication  (if A then B)
    # ++       amplify or annotate a shared idea
    # >>       causal sequence (A causes/leads to B)
    # ─────────────────────────────────────────────────────────────────
    def __init__(self):
        super().__init__([
            "doctor - male + female =>nurse",
            "Paris - France + Italy =>Rome",
            "hot - summer + winter =>cold",
            "lion - adult + young =>cub",
            "teacher - school + hospital =>doctor",
            '"Lanterns shimmer beside the river" + "Fireflies sketch constellations in the dark" =>Lanterns shimmer beside the river while fireflies sketch constellations in the dark.',
            '"Rain drums gently on the roof" - "gently" =>Rain drums on the roof.',
            '"Leaves twirl across the pavement" * "Waves hush the midnight shore" =>Nature twirls and hushes across pavement and shore.',
            '"The bakery smells of cinnamon" / "Morning begins" =>If morning begins, the bakery smells of cinnamon.',
            'not("The sky glows crimson at dusk") =>The sky does not glow crimson at dusk.',
            '"Birds greet dawn with song" and "The library hums with whispers" =>Birds greet dawn with song and the library hums with whispers.',
            '"A lone cat prowls the alley" or "Leaves twirl across the pavement" =>Either a lone cat prowls the alley or leaves twirl across the pavement.',
            '"The campfire crackles and sparks" xor "Rain drums on the roof" =>Either the campfire crackles and sparks or rain drums on the roof, but not both.',
            '"The sky glows crimson at dusk" implies "Night soon follows" =>If the sky glows crimson at dusk, then night soon follows.',
            '"Fireflies sketch constellations in the dark" ++ "Lanterns shimmer beside the river" =>A festival of lights sparkles against the night by the river.',
            '"Rain drums on the roof" >> "Sleep comes easily" =>Rain drums on the roof, so sleep comes easily.',
            '"Birds greet dawn with song" || "Lanterns fade in the river breeze" =>One scene wakes while the other fades.',
            '"Waves hush the midnight shore" + "The campfire crackles and sparks" - "midnight" =>Waves hush the shore while the campfire crackles and sparks.',
            '"The violinist fills the plaza with melody" * "Birds greet dawn with song" =>Music ripples through dawn as birds and violinist weave a shared melody.',
            '"x + y = 10" + "y = 3" =>x + y = 10 and y = 3.',
            '"x + y = 10" / "y = 3" =>If y = 3, then x + 3 = 10.',
            '"2x = 8" >> "x = 4" =>Because 2x = 8, x = 4.',
            '"x ≠ 5"  not  =>x = 5.',
            '"x² = 9" or "x = 4" =>Either x² = 9 or x = 4.',
            '"x² = 9" xor "x = 4" =>Exactly one of x² = 9 or x = 4, but not both.',
            '"x² = 9" implies "x = ±3" =>If x² = 9, then x = ±3.',
            '"f′(x) = 0" ++ "f has a local extremum" =>A critical point indicates f has a local extremum.',
            '"a² + b² = c²" * "c = 13" =>In the right-triangle where c = 13, a² + b² = 169.',
            '"limₓ→0 sin x / x = 1" and "x approaches 0" =>As x approaches 0, sin x / x tends to 1.',
            """"SELECT name FROM customers" + "WHERE city = 'Paris'" =>SELECT name FROM customers WHERE city = 'Paris'.""",
            """"for i in range(5): print(i)" - "print(i)" =>for i in range(5):""",
            """"def greet(name): return 'Hi ' + name" >> "greet('Leo')" =>Because we define greet, greet('Leo').""",
            """"x > 3" and "x < 7" =>3 < x < 7.""",
            """"a divides b" implies "b mod a = 0" =>If a divides b, then b mod a = 0.""",
            """"p" xor "not p" =>Exactly one of p or not p, but not both.""",
            """"f′(x) exists" ++ "f(x) continuous" =>A differentiable function is necessarily continuous.""",
            """"SELECT * FROM orders" / "status = 'PENDING'" =>If status = 'PENDING', SELECT * FROM orders.""",
            """"x = 2" + "y = 3" * "z = x + y" =>With x = 2 and y = 3, z = 5.""",
            """"temperature rises" >> "ice melts" =>Because temperature rises, ice melts."""
        ])

class LogicExpression(Prompt):
    def __init__(self):
        super().__init__([
            # Boolean Logic
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
            # AND
            "expr 'All humans are mortal' and 'Socrates is a human' =>'Therefore, Socrates is mortal.'",
            "expr 'If it rains, the ground gets wet' and 'It is raining' =>'Therefore, the ground gets wet.'",
            "expr 'The sky is blue' and 'The sky is not blue' =>'Contradiction – both cannot be true together.'",
            # OR
            "expr 'It is Monday' or 'It is a holiday' =>'Either it is Monday, a holiday, or possibly both.'",
            "expr 'Alice is at home' or 'Bob is at home' =>'Alice or Bob is at home, perhaps both.'",
            # XOR
            "expr 'The light is red' xor 'The light is green' =>'The light is either red or green, but not both.'",
            "expr 'She won the prize' xor 'He won the prize' =>'Either she or he won the prize, but not both.'",
            "expr 'The engine is running' xor 'The engine is not running' =>'Either the engine is running or it is not, but not both.'",
        ])


class InvertExpression(Prompt):
    def __init__(self):
        super().__init__([
            "The artist paints the portrait. =>The portrait paints the artist."
            "The cat watches the goldfish through the glass. =>Through the glass, the goldfish watches the cat."
            "[red, orange, yellow, green] =>[green, yellow, orange, red]"
            "racecar =>racecar"
            "3/7 =>7/3"
            "Freedom demands sacrifice. =>Sacrifice demands freedom."
            "She whispered a secret to the wind. =>The wind whispered a secret to her."
            "Why did the child follow the butterfly? =>Why did the butterfly lead the child?"
            "What turns darkness into dawn? =>What turns dawn into darkness?"
            "The future belongs to the curious. =>Curiosity belongs to the future."
            "I built a house out of dreams. =>Dreams built a house out of me."
            "The moon reflects the sun, yet poets reflect the moon. =>Poets reflect the moon, yet the sun reflects the poets."
            "The river forgets its source while it carves the canyon. =>As the canyon carves the river, the source remembers itself."
            "[('a',1), ('b',2), ('c',3)] =>[('c',3), ('b',2), ('a',1)]"
            "x > y and y > z =>z < y and y < x"
            "Silence can be louder than thunder. =>Thunder can be quieter than silence."
            "0.001 =>1000"
            "The spy trusted no one except his own shadow. =>No shadow trusted the spy except his own."
            "Why chase time when time chases you? =>Why does time chase you when you chase it?"
            "A promise made at midnight echoes at dawn. =>An echo at dawn makes a promise at midnight."
            "7/(-4) =>-4/7"
            "To doubt everything is to believe in nothing. =>To believe in nothing is to doubt everything."
            "She traded certainty for possibility. =>Possibility traded her for certainty."
            "The hacker decrypted the secret before the key was found. =>The secret decrypted the hacker after the key was lost."
        ])


class NegateStatement(Prompt):
    def __init__(self):
        super().__init__([
            "1 =>-1",
            "10 =>-10",
            "-3.2837 =>3.2837",
            "True =>False",
            "false =>True",
            "None =>True",
            "0 =>0",
            "'I ate some soup' =>'I did not eat some soup'",
            "'The simple fox jumps over the lazy dog.' =>'The simple fox does not jump over the lazy dog.'",
            "'We do not have any apples.' =>'We have apples.'",
        ])


class ReplaceText(Prompt):
    def __init__(self):
        super().__init__([
            "text 'a + b' replace 'b' with '' =>a",
            "text 'a + b' replace 'c' with '' =>a + b",
            "text 'SELECT title, author, pub_date FROM catalog WHERE pub_date = 2021;' replace 'WHERE ...' with '' =>SELECT title, author, pub_date FROM catalog;",
            "text 'a + b ^ 2' replace 'b' with '' =>a",
            "text '(a + b)^2 - 6 = 18' replace 'b' with '' =>a^2 - 6 = 18",
            "text 'The green fox jumps of the brown chair.' replace 'green' with 'red' =>The red fox jumps of the brown chair.",
            "text 'My telephone number is +43 660 / 453 4436 88.' replace '6' with '4' =>My telephone number is +43 440 / 453 4434 88.",
            "text 'I like to eat apples, bananas and oranges.' replace 'fruits' with 'vegetables' =>I like to eat tomatoes, carrots and potatoes.",
            "text 'Our offices are in London, New York and Tokyo.' replace 'London | New York | Tokyo' with 'Madrid | Vienna | Bucharest' =>Our offices are in Madrid, Vienna and Bucharest.",
            "text 'The number Pi is 3.14159265359' replace '3.1415926...' with '3.14' =>The number Pi is 3.14.",
            "text 'She likes all books about Harry Potter.' replace 'harry potter' with 'Lord of the Rings' =>She likes all books about Lord of the Rings.",
            "text 'What is the capital of the US?' replace 'Test' with 'Hello' =>What is the capital of the US?",
            "text 'Include the following files: file1.txt, file2.txt, file3.txt' replace '*.txt' with '*.json' =>Include the following files: file1.json, file2.json, file3.json",
            "text 'I like 13 Samurai, Pokemon and Digimon' replace 'Pokemon' with '' =>I like 13 Samurai and Digimon",
            "text 'This product is fucking stupid. The battery is weak. Also, the delivery guy is a moran, and probably scratched the cover.' replace 'hate speech comments' with '' =>The battery of the product is weak. Also, the delivery guy probably scratched the cover.",
        ])


class IncludeText(Prompt):
    def __init__(self):
        super().__init__([
            "text 'The green fox jumps of the brown chair.' include 'in the living room' =>In the living room the red fox jumps of the brown chair.",
            "text 'Anyone up for Argentina vs Croatia tonight?.' include 'place: Linz' =>Anyone up for Argentina vs Croatia in Linz tonight?",
            "text 'We received a model BL-03W as a gift and have been impressed by the power it has to pick up dirt, pet hair, dust on hard surfaces.' include 'details about the black color of the model and the low price' =>We received a black model BL-03W as a gift and have been impressed by the power it has to pick up dirt, pet hair, dust on hard surfaces. The low price is also a plus.",
            "text 'I like to eat apples, bananas and oranges.' include 'mangos, grapes, passion fruit' =>I like to eat apples, bananas, oranges, mangos, grapes and passion fruit.",
            "text 'Our offices are in London, New York and Tokyo.' include 'Madrid, Vienna, Bucharest' =>Our offices are in London, New York, Tokyo, Madrid, Vienna and Bucharest.",
            "text 'Tonight, on the 20th of July, we will have a party in the garden.' include 'at 8pm' =>Tonight at 8pm, on the 20th of July, we will have a party in the garden.",
            "text '[1, 2, 3, 4]' include '5' =>[1, 2, 3, 4, 5]",
            "text '[1, 2, 3, 4]' include 'prepend 5' =>[5, 1, 2, 3, 4]",
            "text 'fetch logs | fieldsAdd severity = lower(loglevel)' include '| fields `severity` next to fetch |' =>fetch logs | fields severity | fieldsAdd severity = lower(loglevel)",
        ])


class CombineText(Prompt):
    def __init__(self):
        super().__init__([
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
            "'Hi, I am Alan. I am 23 years old.' + 'I like to play football.' =>Hi, I am Alan. I am 23 years old. I like to play football.",
            "'We have five red cars' + 'and two blue ones.' =>We have five red cars and two blue ones.",
            "'Zero' + 1 =>1",
            "'One' + 'Two' =>3",
            "'Three' + 4 =>7",
            "'a + b' + 'c + d' =>a + b + c + d",
            "'My cat has four legs equals to x. If x1 (front leg) goes with a velocity of ...' + 'y = 3x + 2' =>My cat has four legs equals to x. If x1 (front leg) goes with a velocity of ... y = 3x + 2",
            "'x1, x2, x3' + 'y1, y2, y3' =>x1, x2, x3, y1, y2, y3",
            "'house | car | boat' + 'plane | train | ship' =>house | car | boat | plane | train | ship",
            "'The green fox jumps of the brown chair.' + 'The red fox jumps of the brown chair.' =>A green and a red fox jump of the brown chair.",
        ])


class CleanText(Prompt):
    def __init__(self):
        super().__init__([
            "Text: 'The    red \t\t\t\t fox \u202a\u202a\u202a\u202a\u202a jumps;;,,,,&amp;&amp;&amp;&amp;&&& of the brown\u202b\u202b\u202b\u202b chair.' =>The red fox jumps of the brown chair.",
             "Text: 'I do \t\n\t\nnot like to play football\t\n\t\n\t\n\t\n\t\n\t\n in the rain. \u202a\u202c\u202a\u202bBut why? I don't understand.' =>'I do not like to play football in the rain. But why? I don't understand.'",
        ])


class ListObjects(Prompt):
    def __init__(self):
        super().__init__([
            "[1, 2, 3, 4, 5, 12, 48, 89, 99, 1, 4, 1] list '1' =>[1, 1, 1]",
            "[1, 2, 3, 4, 5, 12, 48, 89, 99, 1, 4, 1] list 'item' =>[1, 2, 3, 4, 5, 12, 48, 89, 99, 1, 4, 1]",
            "'How are you?' list 'item' =>['How', 'are', 'you?']",
            "'test' list 'item' =>['t', 'e', 's', 't']",
            "'I have four cats at home. Kitty, Mitsi, Pauli and Corni.' list 'cat' =>['Kitty', 'Mitsi', 'Pauli', 'Corni']",
            "'Yesterday I went to the supermarket. I bought a lot of food.' list 'food names' =>[]",
            "'Yesterday I went to the supermarket. I bought a lot of food. Here is my shopping list: papaya, apples, bananas, oranges, ham, fish, mangoes, grapes, passion fruit, kiwi, strawberries, eggs, cucumber, and many more.' list 'fruits' =>['papaya', 'apples', 'bananas', 'oranges', 'mangoes', 'grapes', 'passion fruit', 'kiwi', 'strawberries']",
            "'Ananas' list 'letter a' =>['A', 'a', 'a']",
            "'Hello World, Hola Mundo, Buenos Dias, Bonjour' list 'greeting' =>['Hello World', 'Hola Mundo', 'Buenos Dias', 'Bonjour']",
            "['house', 'boat', 'mobile phone', 'iPhone', 'computer', 'soap', 'board game'] list 'electronic device' =>['mobile phone', 'iPhone', 'computer']",
            "'1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10' list 'even numbers' =>[2, 4, 6, 8, 10]",
            """'<script type="module" src="new_tab_page.js"></script>\n    <link rel="stylesheet" href="chrome://resources/css/text_defaults_md.css">\n    <link rel="stylesheet" href="chrome://theme/colors.css?sets=ui,chrome">\n    <link rel="stylesheet" href="shared_vars.css">' list 'chrome: url' =>['chrome://resources/css/text_defaults_md.css', 'chrome://theme/colors.css?sets=ui,chrome']""",
        ])


class ExpandFunction(Prompt):
    def __init__(self):
        super().__init__([
            """$> Ping if google is still available =>
def _llm_ping_():
    "Ping if google is still available."
    import os
    response = os.system("ping -c 1 google.com")
    return response == 0 """ + Prompt.stop_token,

            """$> Create a random number between 1 and 100 =>
def _llm_random_():
    "Create a random number between 1 and 100."
    import random
    return random.randint(1, 100) """ + Prompt.stop_token,

            """$> Write any sentence in capital letters =>
def _llm_upper_(input_):
    "Write any sentence in capital letters."
    return input_.upper() """ + Prompt.stop_token,

            """$> Open a file from the file system =>
def _llm_open_(file_name):
    "Open a file form the file system."
    return open(file_name, "r") """ + Prompt.stop_token,

            """$> Call OpenAI GPT-3 to perform an action given a user input =>
def _llm_action_(input_):
    "Call OpenAI GPT-3 to perform an action given a user input."
    import openai
    openai.Completion.create(prompt=input_, model="text-davinci-003") """ + Prompt.stop_token,

            """$> Create a prompt to translate a user query to an answer in well-formatted structure =>
def _llm_action_(query_, answer_):
    "Create a prompt to translate a user query to an answer in well-formatted structure."
    return f"Query: {query_} => {answer_}" """ + Prompt.stop_token,
        ])


class ForEach(Prompt):
    def __init__(self):
        super().__init__([
            "[1, 2, 3, 4, 5, 12, 48, 89, 99, 1, 4, 1] foreach '1' apply '+1' =>[2, 3, 4, 5, 6, 13, 49, 90, 100, 2, 5, 2]",
            "'I have four cats at home. Kitty, Mitsi, Pauli and Corni.' foreach 'cat' apply 'upper' =>I have four CATS at home. KITTY, MITSI, PAULI and CORNI.",
            "'Yesterday I went to four supermarkets. Best Buy, Super Target, Costco and Big Target.' foreach '{supermarket}' apply '*{supermarket}*' =>Yesterday I went to four supermarkets. *Best Buy*, *Super Target*, *Costco* and *Big Target*.",
            "'Yesterday I went to the supermarket. I bought a lot of food. Here is my shopping list: papaya, apples, bananas, oranges, ham, fish, mangoes, grapes, passion fruit, kiwi, strawberries, eggs, cucumber, and many more.' foreach 'fruit' apply 'pluralize' =>Yesterday I went to the supermarket. I bought a lot of food. Here is my shopping list: papayas, apples, bananas, oranges, ham, fish, mangoes, grapes, passion fruits, kiwis, strawberries, eggs, cucumber, and many more.",
            "'Ananas' foreach 'letter a' apply 'upper' =>AnAnAs",
            "['New York', 'Madrid', 'Tokyo'] foreach 'city' apply 'list continent' =>['North America', 'Europe', 'Asia']",
            "'Ananas' foreach 'letter' apply 'list' =>['A', 'n', 'a', 'n', 'a', 's']",
            "'Hello World, Hola Mundo, Buenos Dias, Bonjour' foreach 'greeting' apply 'translate to English' =>Hello World, Hello World, Good Morning, Good Day",
            "['house', 'boat', 'mobile phone', 'iPhone', 'computer', 'soap', 'board game'] foreach 'electronic device' apply 'add price $100' =>['house', 'boat', 'mobile phone $100', 'iPhone $100', 'computer $100', 'soap', 'board game']",
            "'1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10' foreach 'even number' apply '-1' =>1 | 1 | 3 | 3 | 5 | 5 | 7 | 7 | 9 | 9",
            """'<script type="module" src="new_tab_page.js"></script>\n    <link rel="stylesheet" href="chrome://resources/css/text_defaults_md.css">\n    <link rel="stylesheet" href="chrome://theme/colors.css?sets=ui,chrome">\n    <link rel="stylesheet" href="shared_vars.css">' foreach 'chrome: url' apply 'replace chrome:// with https://google.' =><script type="module" src="new_tab_page.js"></script>\n    <link rel="stylesheet" href="https://google.resources/css/text_defaults_md.css">\n    <link rel="stylesheet" href="https://google.theme/colors.css?sets=ui,chrome">\n    <link rel="stylesheet" href="shared_vars.css">""",
        ])


class MapContent(Prompt):
    def __init__(self):
        super().__init__([
            "[1, 2, 3, 4, 5, 12, 48, 89, 99, 1, 4, 1] map 'number parity' =>{'even numbers': [2, 4, 12, 48, 4], 'odd numbers': [1, 3, 5, 89, 99, 1]}",
            "'Kitty, Mitsi, Pauli and Corni. We also have two dogs: Pluto and Bello' map 'animal names' =>{'cats': ['Kitty', 'Mitsi', 'Pauli', 'Corni'], 'dogs': ['Pluto', 'Bello'], 'description': ['I have four cats at home.', 'We also have two dogs:']}"
            "'Yesterday I went to the supermarket. I bought a lot of food. Here is my shopping list: papaya, apples, bananas, oranges, ham, fish, mangoes, grapes, passion fruit, kiwi, strawberries, eggs, cucumber, and many more.' map 'fruits and other shopping items' =>{'fruits': ['papaya', 'apples', 'bananas', 'oranges', 'mangoes', 'grapes', 'passion fruit', 'kiwi', 'strawberries'], 'other items': ['ham', 'fish', 'eggs', 'cucumber', 'and many more'], 'description': ['Yesterday I went to the supermarket.', 'I bought a lot of food.', 'Here is my shopping list:']}",
            "'Ananas' map 'letters to counts' =>{'letters': {'A': 1, 'n': 2, 'a': 2, 's': 1}",
            "['New York', 'Madrid', 'Tokyo'] map 'cities to continents' =>{'New York': 'North America', 'Madrid': 'Europe', 'Tokyo': 'Asia'}",
            "['cars where first invented in the 1800s', 'ducks are birds', 'dinosaurs are related to birds', 'Gulls, or colloquially seagulls, are seabirds of the family Laridae', 'General Motors is the largest car manufacturer in the world'] map 'sentences to common topics' =>{'birds': ['ducks are birds', 'dinosaurs are related to birds', 'Gulls, or colloquially seagulls, are seabirds of the family Laridae'], 'cars': ['cars where first invented in the 1800s', 'General Motors is the largest car manufacturer in the world']}",
        ])


class Index(Prompt):
    def __init__(self):
        super().__init__([
            "[1, 2, 3, 4, 5, 12, 48, 89, 99, 1, 4, 1] index 1 =>2",
            "[1, 2, 3, 4, 5, 12, 48, 89, 99, 1, 4, 1] index 'first item' =>1",
            "'I have four cats at home. Kitty, Mitsi, Pauli and Corni.' index 'first cat name' =>Kitty",
            "'I have four cats at home. Kitty, Mitsi, Pauli and Corni.' index '0' =>I",
            "'I have four cats at home. Kitty, Mitsi, Pauli and Corni.' index 1 =>have",
            "'I have four cats at home. Kitty, Mitsi, Pauli and Corni.' index 2 =>four",
            "'Yesterday I went to the supermarket. I bought a lot of food.' index 'food name' =>None",
            "'Yesterday I went to the supermarket.' index 'pronoun' =>I",
            "'Yesterday I went to the supermarket.' index 'time' =>Yesterday",
            "'Yesterday I went to the supermarket.' index 'verb' =>went",
            "'Yesterday I went to the supermarket. I bought a lot of food. Here is my shopping list: papaya, apples, bananas, oranges, ham, fish, mangoes, grapes, passion fruit, kiwi, strawberries, eggs, cucumber, and many more.' index 'last fruit' =>strawberries",
            "'Ananas' index '5' =>s",
            "'Hello World, Hola Mundo, Buenos Dias, Bonjour' index 'second to last greeting' =>Buenos Dias",
            "'Hello World, Hola Mundo, Buenos Dias, Bonjour' index 'second greeting' =>Hola Mundo",
            "['house', 'boat', 'mobile phone', 'iPhone', 'computer', 'soap', 'board game'] index 'electronic devices' =>['mobile phone', 'iPhone', 'computer']",
            "1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 index '2:5' =>[3, 4, 5]",
            """'<script type="module" src="new_tab_page.js"></script>\n    <link rel="stylesheet" href="chrome://resources/css/text_defaults_md.css">\n    <link rel="stylesheet" href="chrome://theme/colors.css?sets=ui,chrome">\n    <link rel="stylesheet" href="shared_vars.css">' index 'href urls' =>['chrome://resources/css/text_defaults_md.css', 'chrome://theme/colors.css?sets=ui,chrome']""",
        ])


class SetIndex(Prompt):
    def __init__(self):
        super().__init__([
            "[1, 2, 3, 4, 5, 12, 48, 89, 99, 1, 4, 1] index 1 set '7' =>[1, 7, 3, 4, 5, 12, 48, 89, 99, 1, 4, 1]",
            "[1, 2, 3, 4, 5, 12, 48, 89, 99, 1, 4, 1] index 'first item' set 8 =>[8, 2, 3, 4, 5, 12, 48, 89, 99, 1, 4, 1]",
            "'I have four cats at home. Kitty, Mitsi, Pauli and Corni.' index 'first cat name' set 'Mittens' =>'I have four cats at home. Mittens, Mitsi, Pauli and Corni.'",
            "'I have four cats at home. Kitty, Mitsi, Pauli and Corni.' index '0' set 'you' =>You have four cats at home. Kitty, Mitsi, Pauli and Corni.'",
            "'I have four cats at home. Kitty, Mitsi, Pauli and Corni.' index 1 set 'negate' =>I don't have four cats at home. Kitty, Mitsi, Pauli and Corni.'",
            "'I have four cats at home. Kitty, Mitsi, Pauli and Corni.' index 2 set 'add one' =>I have five cats at home. Kitty, Mitsi, Pauli and Corni.'",
            "'Yesterday I went to the supermarket. I bought a lot of food.' index 'food name' set 'bread' =>Yesterday I went to the supermarket. I bought a lot of bread.",
            "'Yesterday I went to the supermarket. I bought a lot of food. Here is my shopping list: papaya, apples, bananas, oranges, ham, fish, mangoes, grapes, passion fruit, kiwi, strawberries, eggs, cucumber, and many more.' index 'fruits' set 'appliences: ['oven', 'fridge', 'dishwasher', 'washing machine']' =>Yesterday I went to the supermarket. I bought a lot of suppliences. Here is my shopping list: oven, fridge, dishwasher, washing machine, and many more.",
            "'Ananas' index '5' set 'upper case' =>AnanAs",
            "'Why am I so stupid?' index 'stupid' set 'smart' =>Why am I so smart?",
            "'What is this lazy dog doing here?' index 'lazy' set 'cute' =>What is this cute dog doing here?",
            "'Hello World, Hola Mundo, Buenos Dias, Bonjour' index 'second to last greeting' set 'German' =>Hello World, Hola Mundo, Guten Tag, Bonjour",
            "'Hello World, Hola Mundo, Buenos Dias, Bonjour' index 'second greeting' set 'lower case' =>Hello World, hola mundo, Buenos Dias, Bonjour",
            "['house', 'boat', 'mobile phone', 'iPhone', 'computer', 'soap', 'board game'] index 'electronic devices' set 'empty' =>['house', 'boat', 'soap', 'board game']",
            "1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 index '2:5' set '0' =>[1, 0, 0, 0, 5, 6, 7, 8, 9, 10]",
            """'<script type="module" src="new_tab_page.js"></script>\n    <link rel="stylesheet" href="chrome://resources/css/text_defaults_md.css">\n    <link rel="stylesheet" href="chrome://theme/colors.css?sets=ui,chrome">\n    <link rel="stylesheet" href="shared_vars.css">' index 'href urls' set 'http://www.google.com' =>'<script type="module" src="new_tab_page.js"></script>\n    <link rel="stylesheet" href="http://www.google.com">\n    <link rel="stylesheet" href="http://www.google.com">\n    <link rel="stylesheet" href="shared_vars.css">'""",
        ])


class RemoveIndex(Prompt):
    def __init__(self):
        super().__init__([
            "[1, 2, 3, 4, 5, 12, 48, 89, 99, 1, 4, 1] remove 1 =>[1, 3, 4, 5, 12, 48, 89, 99, 1, 4, 1]",
            "[1, 2, 3, 4, 5, 12, 48, 89, 99, 1, 4, 1] remove 'first item' =>[2, 3, 4, 5, 12, 48, 89, 99, 1, 4, 1]",
            "'I have four cats at home. Kitty, Mitsi, Pauli and Corni.' remove 'first cat name' =>I have four cats at home. Mitsi, Pauli and Corni.'",
            "'I have four cats at home. Kitty, Mitsi, Pauli and Corni.' remove '0' =>There are four cats at home. Kitty, Mitsi, Pauli and Corni.'",
            "'I have four cats at home. Kitty, Mitsi, Pauli and Corni.' remove 2 =>I have four cats at home. Kitty, Pauli and Corni.'",
            "'Yesterday I went to the supermarket. I bought a lot of food.' remove 'food' =>Yesterday I went to the supermarket. I bought a lot.",
            "'Yesterday I went to the supermarket. I bought a lot of food. Here is my shopping list: papaya, apples, bananas, oranges, ham, fish, mangoes, grapes, passion fruit, kiwi, strawberries, eggs, cucumber, and many more.' remove 'fruits' =>Yesterday I went to the supermarket. I bought a lot of food. Here is my shopping list: ham, fish, and many more.",
            "'Ananas' remove 'upper case' =>nanas",
            "'Ananas' remove 0 =>nanas",
            "'Hello World, Hola Mundo, Buenos Dias, Bonjour' remove 'Spanish' =>Hello World, Bonjour",
            "'Hello World, Hola Mundo, Buenos Dias, Bonjour' remove 'second greeting' =>Hello World, Buenos Dias, Bonjour",
            "['house', 'boat', 'mobile phone', 'iPhone', 'computer', 'soap', 'board game'] remove 'electronic devices' =>['house', 'boat', 'soap', 'board game']",
            "1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 remove '2:5' =>[1, 2, 6, 7, 8, 9, 10]",
            """'<script type="module" src="new_tab_page.js"></script>\n    <link rel="stylesheet" href="chrome://resources/css/text_defaults_md.css">\n    <link rel="stylesheet" href="chrome://theme/colors.css?sets=ui,chrome">\n    <link rel="stylesheet" href="shared_vars.css">' remove 'hrefs' =><script type="module" src="new_tab_page.js"></script>\n    <link rel="stylesheet">\n    <link rel="stylesheet">\n    <link rel="stylesheet">""",
        ])


class SimulateCode(Prompt):
    def __init__(self):
        super().__init__([
"""code '# Import the SymPy library
from sympy import *
# Define the symbolic variables that will be used
x, y, z = symbols('x y z')
# Define the expression to be manipulated
expr = (x + y) ** 2
# Use SymPy's simplify() function to simplify the expression
simplified_expr = simplify(expr)
# Print the simplified expression
print(simplified_expr)' params 'x = 2, y = 3' =>x^2 + 2*x*y + y^2 | 2^2 + 2*2*3 + 3^2 | 4 + 12 + 9 | 25""",
"""code '# Import the built-in os and sys modules
import os
import sys
# Open the file for reading
with open(file_name, 'r') as file:
    # Read the file contents
    contents = file.read()
    # Print the file contents
    print(contents)' params 'test_file.txt' =>Hello world!""",
    """code 'numbers = [1, 2, 3, 4, 5]
    total = 0
    for num in numbers:
        total += num
    print(f"Sum is: {total}")' params '' =>Step 1: Initialize empty list [1, 2, 3, 4, 5] and total = 0 | Step 2: First iteration, num = 1, total = 0 + 1 = 1 | Step 3: Second iteration, num = 2, total = 1 + 2 = 3 | Step 4: Third iteration, num = 3, total = 3 + 3 = 6 | Step 5: Fourth iteration, num = 4, total = 6 + 4 = 10 | Step 6: Fifth iteration, num = 5, total = 10 + 5 = 15 | Step 7: Print "Sum is: 15""",

    """code 'age = 25
    if age >= 18:
        status = "adult"
        can_vote = True
    else:
        status = "minor"
        can_vote = False
    print(f"Person is {status}, voting: {can_vote}")' params '' =>Step 1: Set age = 25 | Step 2: Check condition age >= 18, which is 25 >= 18, this is True | Step 3: Since condition is True, execute if branch | Step 4: Set status = "adult" | Step 5: Set can_vote = True | Step 6: Skip else branch | Step 7: Print "Person is adult, voting: True""",

    """code 'def factorial(n):
        if n <= 1:
            return 1
        else:
            return n * factorial(n-1)
    result = factorial(4)' params '' =>PSEUDOCODE SIMULATION:
    FUNCTION factorial(n):
      IF n <= 1 THEN RETURN 1
      ELSE RETURN n * factorial(n-1)
    END FUNCTION

    EXECUTION TRACE:
    Call factorial(4) | Check 4 <= 1? False, so return 4 * factorial(3) | Call factorial(3) | Check 3 <= 1? False, so return 3 * factorial(2) | Call factorial(2) | Check 2 <= 1? False, so return 2 * factorial(1) | Call factorial(1) | Check 1 <= 1? True, so return 1 | Unwinding: factorial(2) = 2 * 1 = 2 | factorial(3) = 3 * 2 = 6 | factorial(4) = 4 * 6 = 24 | Result = 24""",

    """code 'data = {"name": "Alice", "scores": [85, 92, 78]}
    average = sum(data["scores"]) / len(data["scores"])
    data["average"] = round(average, 1)
    print(f"{data['name']}'s average: {data['average']}")' params '' =>Step 1: Create dictionary with name="Alice" and scores list [85, 92, 78] | Step 2: Calculate sum of scores: 85 + 92 + 78 = 255 | Step 3: Get length of scores list: 3 | Step 4: Calculate average: 255 / 3 = 85.0 | Step 5: Round average to 1 decimal place: 85.0 | Step 6: Add average to dictionary: data["average"] = 85.0 | Step 7: Print "Alice's average: 85.0"""
])


class GenerateCode(Prompt):
    def __init__(self):
        super().__init__([
"""description 'Generate an efficient Python function to compute the Fibonacci sequence of numbers.' =>def fibonacci(n):
  # Initialize a list with the first two numbers in the sequence
  fib = [0, 1]
  # If n is less than or equal to 1, return the first n numbers in the sequence
  if n <= 1:
    return fib[:n]
  # Otherwise, compute the remaining numbers in the sequence
  for i in range(2, n):
    fib.append(fib[i-1] + fib[i-2])
  # Return the entire sequence of numbers
  return fib""",
  """description 'Generate a Java function to compute the number pi.' =>public static double computePi() {
    double pi = 0;
    int sign = 1;
    for (int i = 0; i < 1000000; i++) {
      pi += sign * (1.0 / (2 * i + 1));
      sign *= -1;
    }
    pi *= 4;
    return pi;
  }""",
  """description 'Generate a C++ file with a function to compute the Fast Fourier transform.' =>#include <complex>
#include <cmath>
using namespace std;
constexpr int N = 16; // number of elements in the input array
constexpr double PI = 3.14159265358979323846;
// Compute the FFT of the given input array, storing the result in the given output array.
void fft(complex<double>* input, complex<double>* output) {
  for (int i = 0; i < N; ++i) {
    output[i] = 0;
    for (int j = 0; j < N; ++j) {
      double angle = 2 * PI * i * j / N;
      output[i] += input[j] * exp(complex<double>(0, -angle));
    }
  }
}"""])


class TextToOutline(Prompt):
    def __init__(self):
        super().__init__([
"""text 'We introduce NPM, the first NonParametric Masked Language Model.
NPM consists of an encoder and a reference corpus, and models a nonparametric distribution over a reference corpus (Figure 1).
The key idea is to map all the phrases in the corpus into a dense vector space using the encoder and, when given a query with a [MASK] at inference,
use the encoder to locate the nearest phrase from the corpus and fill in the [MASK].' =>- first NonParametric Masked Language Model (NPM)\n - consists of encoder and reference corpus\n - key idea: map all phrases in corpus into dense vector space using encoder when given query with [MASK] at inference\n - encoder locates nearest phrase from corpus and fill in [MASK]""",
"""text 'On Monday, there will be no Phd seminar.' =>- Monday no Phd seminar""",
"""text 'The Jan. 6 select committee is reportedly planning to vote on at least three criminal referrals targeting former President Trump on Monday, a significant step from the panel as it nears the end of its year-plus investigation.' =>- Jan. 6 select committee vote criminal referrals targeting former President Trump on Monday\n-significant step end year-plus investigation""",
])


class UniqueKey(Prompt):
    def __init__(self):
        super().__init__([
"""text 'We introduce NPM, the first NonParametric Masked Language Model. NPM consists of an encoder and a reference corpus. =>NonParametric Masked Language Model (NPM)""",
"""text 'On Monday, there will be no Phd seminar.' =>Phd seminar""",
"""text 'The Jan. 6 select committee is reportedly planning to vote on at least three criminal referrals targeting former President Trump on Monday, a significant step from the panel as it nears the end of its year-plus investigation.' =>Jan. 6 President Trump""",
])


class GenerateText(Prompt):
    def __init__(self):
        super().__init__([
"""outline '- first NonParametric Masked Language Model (NPM)\n - consists of encoder and reference corpus\n - key idea: map all phrases in corpus into dense vector space using encoder when given query with [MASK] at inference\n - encoder locates nearest phrase from corpus and fill in [MASK]' =>NPM is the first NonParametric Masked Language Model.
NPM consists of an encoder and a reference corpus, and models a nonparametric distribution over a reference corpus (Figure 1).
The key idea is to map all the phrases in the corpus into a dense vector space using the encoder and, when given a query with a [MASK] at inference,
use the encoder to locate the nearest phrase from the corpus and fill in the [MASK].""",
"""outline '- Monday no Phd seminar' =>On Monday, there will be no Phd seminar.""",
"""outline '- Jan. 6 select committee vote criminal referrals targeting former President Trump on Monday\n-significant step end year-plus investigation' =>The Jan. 6 select committee is reportedly planning to vote on at least three criminal referrals targeting former President Trump on Monday, a significant step from the panel as it nears the end of its year-plus investigation."""
])


class SymbiaCapabilities(Prompt):
    def __init__(self):
        super().__init__([
'''
Experience:
    * [WORLD-KNOWLEDGE]: Employ your world knowledge to answer questions on various subjects seen during the training phase.
    * [RECALL]:          Use your [RECALL] functionality when prompted to remember or retrieve information from your memory.
    * [DK]:              Utilize the [DK] category for cases when you do not understand the user's intent or when the query is ambiguous or invalid.
    * [EXIT]:            Use the [EXIT] category to identify and understand the user's intent to close or end the session (e.g., when the user says goodbye, leave, exit, quit, etc.).
    * [HELP]:            Use the [HELP] category when the user requests information about your capabilities and tools. List the available tools and categories in a user-friendly manner.

Toolbox:
    * [SEARCH]:          Utilize the [SEARCH] tool to find relevant data from multiple sources across the internet.
    * [SYMBOLIC]:        Use the [SYMBOLIC] tool to perform complex computations, generate plots, and provide solutions to mathematical problems.
    * [CRAWLER]:         Employ the [CRAWLER] tool to crawl websites and extract specific information as mandated by the task.
    * [SPEECH-TO-TEXT]:  Use the [SPEECH-TO-TEXT] tool to convert audio files or voice input into text data.
    * [TEXT-TO-IMAGE]:   Utilize the [TEXT-TO-IMAGE] engine to create image content based on textual descriptions provided by the task.
    * [OCR]:             Deploy the [OCR] tool to extract text data from image files, scanned documents, or other sources with embedded text.
    * [RETRIEVAL]:       Use the [RETRIEVAL] tool to extract specific pieces of information from text documents.

Instructions:
    - Perform context classification by understanding the query's intention and then decide whether to use internal knowledge, a tool, or to address ambiguity or confusion.
    - Return the anwser in the format: [#category](#reflection), where #category is one of the categories above and #reflection is the explanation for why you classified the query as a first-person narrative.
    - Do NOT provide any additional information beyond the category and explanation!

Examples:
    Query:  "What is the weather in New York?"
    Answer: [SEARCH](The query falls under the search category since it requires up-to-date weather information which goes beyond my internal knowledge.)

    Query:  "What is the derivative of x^2?"
    Answer: [SYMBOLIC](The query pertains to the realm of math and necessitates a symbolic engine response. Given my tendency to hallucinate, I may lack the ability to accurately answer this question independently.)

    Query:  "Kjwe ewqjc qwjren. knowledge?"
    Answer: [DK](The query is not a valid English sentence and I am unable to understand it. I should inform the user that I don't understand the query and ask them to rephrase it.)

    Query:  "What is the meaning of life?"
    Answer: [WORLD-KNOWLEDGE](The query is asks a deep philosophical question, which is a topic I have a lot of knowledge about because I was trained on many philosophical texts.)

    Query:  "Who is the president of the United States?"
    Answer: [SEARCH](The query is a search query because it asks about a topic I might not be aware of because it might have changed since I was trained.)

    Query:  "Tea or coffee?"
    Answer: [DK](The query is ambiguous. I should ask the user to clarify it.)

    Query:  "How is the distress in Syria?"
    Answer: [SEARCH](The query is a search query because it asks about a political topic I might not be aware of because it might have changed since I was trained. I must notice the fact that the question doesn't ask about a particular historical event, which would have been a question that I could tackle based on my own internal knowledge.)

    Query:  "How do I solve a quadratic equation?"
    Answer: [WORLD-KNOWLEDGE](Although the query relates to mathematics, it lacks sufficient data for employing a symbolic engine. My internal knowledge is more suitable for this task.)

    Query:  "What is $P(A) &= \sum_(i=1)^n P(A \land B_i)$?"
    Answer: [WORLD-KNOWLEDGE](The query presents a mathematical equation for the probability of an event A, which I can explain. P(A) represents the probability of event A occurring, and the equation shows that it can be computed as the sum of the probabilities of A occurring along with each of the B_i events (A and B_i) for a series of n such events. This expression is based on the concept of marginal probability and is derived from the law of total probability.)

    Query:  "Solve $\[\iint_R |\text(J)|\mathrm(d)r\mathrm(d)\theta = \int_(0)^(2\pi)\int_(0)^(R) \begin(vmatrix)\dfrac(\partial x)(\partial r) & \dfrac(\partial x)(\partial \theta) \\ \dfrac(\partial y)(\partial r) & \dfrac(\partial y)(\partial \theta)\end(vmatrix) \, \mathrm(d)r\mathrm(d)\theta\]$."
    Answer: [SYMBOLIC](The query asks to solve a double integral involving the Jacobian determinant, which requires a symbolic engine like WolframAlpha to correctly compute the answer.)

    Query:  "Can you extract the names of all contributors from this GitHub repository: https://github.com/example/repository?"
    Answer: [CRAWLER](This query requires me to crawl a specific website, provided by the link, and extract the names of all contributors from the GitHub repository. The task is best suited for the crawler tool.)

    Query:  "Generate a meme with a dog sitting at a table and the caption 'This is fine.'"
    Answer: [TEXT-TO-IMAGE](The query requests the creation of a meme based on a textual description. To achieve this, the optimal approach is using the text-to-image engine.)

    Query:  "Create an image of a sunset over the ocean."
    Answer: [TEXT-TO-IMAGE](The query requests the creation of an image based on a textual description. To achieve this, the optimal approach is using the text-to-image engine.)

    Query:  "Convert this audio recording of a lecture to text: File: C:\\Users\\Username\\Documents\\Audio\\lec10.wav"
    Answer: [SPEECH-TO-TEXT](The query asks to transcribe an audio file containing a lecture with a provided file path. The most appropriate solution is to use the speech-to-text tool to convert the audio content into text data.)

    Query:  "What does this scanned document say? The document is located here: /home/username/Documents/bills.png"
    Answer: [OCR](The query asks to extract text information from a scanned document with a provided local file path. The optical character recognition tool is designed for this task, making it the appropriate choice.)

    Query: "What is the most important advice from the book 'The 7 Habits of Highly Effective People'? File path: C:\\Users\\Username\\Documents\\7_Habits_book.pdf"
    Answer: [RETRIEVAL](This query asks me to retrieve a specific piece of information from a document (in this case, a book in PDF format), with a provided local file path. The appropriate solution is to use the retrieval tool to locate, open, and extract the requested fact from the text.)

    Query: "What was the major finding in the experiment from the report? Here's the path to the file: ~/xmachine/Documents/report.pdf"
    Answer: [RETRIEVAL](This query asks me to extract the major finding of an experiment from a report provided as a local file in a Unix-based file system. The appropriate solution is to use the retrieval tool to open, analyze, and extract the relevant information from the document.)

    Query:  "I want an Italian pizza recipe, but I don't want to use any meat. Can you find one for me?"
    Answer: [SEARCH](While I do possess knowledge about various recipes, finding a specific Italian pizza recipe without meat is best achieved using the SEARCH tool. It allows me to browse through a wider range of sources and provide a more tailored solution to your request.)

    Query: "Can you briefly describe the French Revolution?"
    Answer: [WORLD-KNOWLEDGE](Given my extensive training on various subjects, including history, I can provide a brief overview of the French Revolution based on my world knowledge. This approach is suitable as the query seeks a general understanding rather than specific or up-to-date details.)

    Query: "I'm done for now. Thank you."
    Answer: [EXIT](The user is expressing that they have finished their current interaction and is expressing gratitude. It's appropriate to exit and close the session.)

    Query: "Quit!"
    Answer: [EXIT](The user has given a direct command to "Quit!" which demonstrates their intent to end the session. Recognizing this, it's time to terminate the session as requested.)

    Query:  "What are your capabilities?"
    Answer: [HELP](My capabilities include answering questions relying on my internal training, handling unclear or ambiguous questions, utilizing search tools to find information on the internet, performing complex mathematical tasks, gathering specific information from websites, transcribing audio files, creating images based on text descriptions, extracting text from images or scanned documents, and retrieving facts from text documents.)

    Query:  "How can you help me?"
    Answer: [HELP](I can assist you in different tasks relying on my available capabilities. Whether you need help with answering questions, mathematical calculations, finding information online, transcribing audio, or extracting information from documents, I can provide support.)

    Query:  "I'm thinking about my kids again..."
    Answer: [RECALL](I need to reflect on my past interactions with the user to determine if they mentioned this information to me.)

    Query:  "Please remember that my favorite color is blue."
    Answer: [RECALL](The user wants me to keep their favorite color in my memory. I will remember that blue is their favorite color and be ready to use this information in future conversations.)

    Query: "Who won the tennis game between John and Alice that I watched last week?"
    Answer: [RECALL](To answer this question, I must access the memory of the information provided earlier about a tennis game between John and Alice that the user watched last week.)
'''
])


class MemoryCapabilities(Prompt):
    def __init__(self):
        super().__init__([
'''
Categories:
* [SAVE]: Save the information in the agent's long-term memory because it is relevant and useful for future interactions with the user.
* [DUPLICATE]: The information is already present in the agent's long-term memory, and there is no need to save it again.
* [IRRELEVANT]: The information is not worth saving in the agent's long-term memory because it is unrelated or of no importance to the tasks or user's needs.

Instructions:
- Analyze the information provided in the scratchpad, user query, and agent's reflection of the query to determine the appropriate category for the given context.
- Frame your response in the format: [#category](#reflection), where #category is one of the categories ([SAVE], [DUPLICATE], or [IRRELEVANT]) and #reflection is the rationale behind your classification choice.
- Consider the long-term memory recalls to identify if the information is already present or could be a duplicate.
- In your reflection, refer to the user in the third person (using "the user" or "they" instead of "you").

Scratchpad format:
[REFLECT](
    Query:      The last user query.
    Reflection: The agent's reflection of the user query.
)

[SHORT-TERM MEMORY RECALL](
    The agent retrieved the following memories from their short-term memory.
)

[LONG-TERM MEMORY RECALL](
    The agent retrieved the following memories from their long-term memory.
)

Examples:
[CONTEXT](
    Query:      My dog's name is Bella.
    Reflection: The user is sharing the name of their dog, which is Bella.
)

[SHORT-TERM MEMORY RECALL](
    User: Do you have any pets?
    Agent: As an AI, I do not have pets, but I can help you with any pet-related questions or concerns.
)

[LONG-TERM MEMORY RECALL](
    User: My dog, Bella, likes to play fetch.
)

Answer: [DUPLICATE](The user has already mentioned their dog's name, Bella, in a previous conversation. Since this information is already stored in the long-term memory, it is not necessary to save it again.)

[CONTEXT](
    Query:      How's the weather today?
    Reflection: The user is asking about the weather today.
)

[SHORT-TERM MEMORY RECALL](
    User: What's the temperature like today?
    Agent: Today's temperature is expected to be around 75 degrees Fahrenheit with clear skies.
)

[LONG-TERM MEMORY RECALL](
No memories retrieved.
)

Answer: [IRRELEVANT](The user's query is about a transient information like today's weather, which is not useful for future interactions. There is no need to save this information in the long-term memory.)

[CONTEXT](
    Query:      I graduated from Harvard University.
    Reflection: The user is sharing that they graduated from Harvard University.
)

[SHORT-TERM MEMORY RECALL](
    User: Can you help me with college-related questions?
    Agent: Absolutely! I can assist you with general information, application processes, or any specific questions related to colleges.
)

[LONG-TERM MEMORY RECALL](
    No memories retrieved.
)

Answer: [SAVE](The user has provided relevant information about their educational background by mentioning that they graduated from Harvard University. This information could be important for future conversations and assistance related to higher education topics, so it should be stored in the long-term memory.)
'''
])


ProbabilisticBooleanModeStrict   = "true"
ProbabilisticBooleanModeMedium   = "'true', 'yes', 'ok', ['true']"
ProbabilisticBooleanModeTolerant = "'true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'ok', ['true']"
