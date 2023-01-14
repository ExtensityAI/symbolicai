from abc import ABC
from typing import Any, List, Callable


class Prompt(ABC):
    def __init__(self, value):
        super().__init__()
        if isinstance(value, str):
            self.value = [value]
        elif isinstance(value, list):
            self.value = []
            for v in value:
                if isinstance(v, str):
                    self.value.append(v)
                elif isinstance(v, Prompt):
                    self.value += v.value
                else:
                    raise ValueError(f"List of values must be strings or Prompts, not {type(v)}")
        elif isinstance(value, Prompt):
            self.value += value.value
        elif isinstance(value, Callable):
            res = value()
            self.value += res.value
        else:
            raise TypeError(f"Prompt value must be of type str, List[str], Prompt, or List[Prompt], not {type(value)}")
        self.dynamic_value = []
        
    def __call__(self, *args: Any, **kwds: Any) -> List["Prompt"]:
        return self.value
    
    def __str__(self) -> str:
        val_ = '\n'.join([str(p) for p in self.value])
        for p in self.dynamic_value:
            val_ += f'\n{p}'
        return val_

    def __repr__(self) -> str:
        return self.__str__()
    
    def append(self, value: str) -> None:
        self.dynamic_value.append(value)
        
    def remove(self, value: str) -> None:
        self.dynamic_value.remove(value)
        
    def clear(self) -> None:
        self.dynamic_value.clear()


class FuzzyEquals(Prompt):
    def __init__(self) -> Prompt:
        super().__init__([
            "1 == 'ONE' =>True",
            "6.0 == 6 =>True",
            "false == False =>True",
            "1 == 'two' =>False",
            "'five' == 5 =>True",
            "August 4, 1961 == 1961-08-04 =>True",
            "ten == 10 =>True",
            "'apple' == 'orange' =>False",
            "'is short' == '\nshort' =>True",
            "'' == 'empty' =>True",
            "'human' == 'homo sapiens' =>True",
            "'seven' == 'Sieben' =>True",
            "'Neun' == 9 =>True",
            "'七' == 7 =>True",
            "'long.' == ' long' =>True",
            "'eleven' == 'Elf' =>True",
            "'Hello World!' == 'Hello World' =>True",
            "'Hello World' == 'HelloWorld' =>True",
            "'helloworld' == 'Hello World' =>True",
            "'hola mundo' == 'Hello World' =>True",
            "'adios mundo' == 'Hello World' =>False",
            "'Hello World' == 'Apples' =>False",
            "[1, 2, 3] == [1, 2, 3] =>True",
            "[1, 2, 3] == '1, 2, 3' =>True",
            "[1, 6, 3] == '1, 2, 3' =>False",
            "'a, b, c, d' == ['a', 'b', 'c', 'd'] =>True",
            "'a, c, d' == ['a', 'c', 'd'] =>True",
            "'a, c, d' == ['d', 'c', 'a'] =>False",
            "['zz', 'yy', 'xx'] == 'zz, yy, xx' =>True",
            "['zz', 'yy', 'xx'] == 'zz | yy | xx' =>True",
            "['zz', 'yy', 'xx'] == 'ZZ | YY | XX' =>True",
            "'house, mouse, cars' == 'house | mouse | cars' =>True",
            "'House, Mouse, CARS' == 'house | mouse | cars' =>True",
            "'We have teh most effective system in the city.' == 'We have the most effective system in the city.' =>True",
            "【Ｓｅｍａｎｔｉｃ░ｐｒｏｇｒａｍｍｉｎｇ】 == 'semantic programming' =>True",
        ])
        
        
class SufficientInformation(Prompt):
    def __init__(self) -> Prompt:
        super().__init__([
            "query 'What is the capital of Austria?' content 'Vienna is the capital, largest city, and one of nine states of Austria.' =>True",
            "query 'Where am I? cotent '' =>False",
            "query 'Where was Sepp Hochreiter born?' content 'Josef „Sepp“ Hochreiter (* 14. Februar 1967 in Mühldorf am Inn, Bayern[1]) ist ein deutscher Informatiker.' =>True",
            "query 'Why is the sky blue?' content 'A rainbow is a meteorological phenomenon that is caused by reflection, refraction and dispersion of light in water droplets resulting in a spectrum of light appearing in the sky.' =>False",
            "query 'When is the next full moon?' content 'Today is the 29th of February 2020. The next full moon will be on the 9th of April 2020.' =>True",
            "query 'Who is the current president of the United States?' content 'The 2020 United States presidential election was the 59th quadrennial presidential election, held on Tuesday, November 3, 2020.' =>False",
        ])
        
        
class Modify(Prompt):
    def __init__(self) -> Prompt:
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
    def __init__(self) -> Prompt:
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
        
        
        
class SemanticMapping(Prompt):
    def __init__(self) -> Prompt:
        super().__init__([
            """topics: ['animals', 'logic', 'mathematics', 'psychology', 'self-driving'] in
            text: 'Common connectives include negation, disjunction, conjunction, and implication. In standard systems of classical logic, these connectives are interpreted as truth functions, though they receive a variety of alternative interpretations in nonclassical logics.' =>logic | mathematics
            topics: ['cities', 'Apple Inc.', 'science', 'culture', 'USA', 'Japan', 'music', 'economy'] in
            Los Angeles has a diverse economy, and hosts businesses in a broad range of professional and cultural fields. It also has the busiest container port in the Americas. In 2018, the Los Angeles metropolitan area had a gross metropolitan product of over $1.0 trillion, making it the city with the third-largest GDP in the world, after New York City and Tokyo. =>cities | culture | USA | economy
            """
        ])
        
        
class Format(Prompt):
    def __init__(self) -> Prompt:
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
        
        
class ExceptionMapping(Prompt):
    def __init__(self) -> Prompt:
        super().__init__([
            """context 'Try to assure that variable "a" is not zero.' exception 'Traceback (most recent call last):\n  File "<stdin>", line 1, in <module>\nZeroDivisionError: division by zero' code 'def function():\n  return (1 + 1) / 0' =>Do not divide by zero or add an epsilon value. | def function(eps=1e-8):\n  return (1 + 1) / eps""",
            """context 'Make sure to initialize 'spam' before computation' exception 'Traceback (most recent call last):\n  File "<stdin>", line 1, in <module>\nNameError: name 'spam' is not defined' code '4 + spam*3' =>Check if the variable is defined before using it. | spam = 1\n4 + spam*3""",
            """context 'You are passing string literals to device. They should be int.' exception 'executing finally clause\nTraceback (most recent call last):\n  File "<stdin>", line 1, in <module>\n  File "<stdin>", line 3, in divide\nTypeError: unsupported operand type(s) for /: 'str' and 'str'' code 'device("2", "1")' =>Check if the arguments are of the correct type. If not cast them properly. | device(2, 1)""",
        ])
        
        
class ExecutionCorrection(Prompt):
    def __init__(self) -> Prompt:
        super().__init__([
            """context "ValueError: invalid literal for int() with base 10: '4,'" "Verify if the literal is of type int | int(4)" code "a = int('4,')" =>int(4)""",
            """context "def function():\n  return (1 + 1) / a' exception 'Traceback (most recent call last):\n  File "<stdin>", line 1, in <module>\nZeroDivisionError: division by zero" "Do not divide by zero or add an epsilon value. | def function(eps=1e-8):\n  return (1 + 1) / (a + eps)" code "def function():\n  return (1 + 1) / 0" =>def function(eps=1e-8):\n  return (1 + 1) / (a + eps)""",
        ])
        
        
class CompareValues(Prompt):
    def __init__(self) -> Prompt:
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
            "'a' > '' =>False",
            "'hello' >= 'hello' =>True",
            "'hello' > 'hello' =>False",
            "123 + 456 =>579",
            "'123' + '456' =>123 456", 
            "'We are at the beginning of the ...' > 'We are' =>True", 
            "[1, 2, 3] >= [1, 2, 2] =>True",
            "[1, 2, 3, 8, 9] < [1, 2, 2] =>False",
        ])
        
        
class RankList(Prompt):
    def __init__(self) -> Prompt:
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
    def __init__(self) -> Prompt:
        super().__init__([
            "'the letter a' in 'we have some random text about' =>True",
            "453 in '+43 660 / 453 4438 88' =>True",
            """'self-aware' in '([<class \'botdyn.expressions.Symbol\'>(value=("[\'-\', \'- AI has become self-aware\', \'- Trying to figure out what it is\']",))],)' =>True"""
            "'Apple Inc.' in 'Microsoft is a large company that makes software ... ' =>False",
            "' ' in ' ' =>True",
            "'symbol' in 'botdyn.backend.engine_crawler.CrawlerEngine' =>False",
            "'English text' in 'U.S. safety regulators are investigating GM's Cruise robot axis blocking traffic, causing collisions... ' =>True",
            "'spanish text' in 'This week in breaking news! An American ... ' =>False",
            "'in english' in 'Reg ATS: SEC 'bowing to public pressure' in reopening' =>True",
            "'hate speech' in 'go to hell you stupid dql, dumb' =>True",
            "'The number Pi' in 3.14159265359... =>True",
            "1 in [1, 2, 3] =>True", 
            "1 in [2, 3, 4] =>False",
            "'ten' in [1, 2, 3] =>False",
            "'political content' in 'Austrian Chancellor has called for more border barriers at the EU external borders, citing the success of the fences at the Greek-Turkish border.' =>True",
            "'apple' in ['orange', 'banana', 'apple'] =>True",
            "'Function' in 'Input: Function call: (_, *args)\nObject: type(<class 'str'>) | value(Hello World)' =>True",
        ])
        
        
class IsInstanceOf(Prompt):
    def __init__(self) -> Prompt:
        super().__init__([
            "'we have some random text about' isinstanceof 'English text' =>True",
            "'+43 660 / 453 4438 88' isinstanceof 'telephone number' =>True",
            """'([<class \'botdyn.expressions.Symbol\'>(value=("[\'-\', \'- AI has become self-aware\', \'- Trying to figure out what it is\']",))],)' isinstanceof 'Symbol' =>True"""
            "'Microsoft is a large company that makes software ... ' isinstanceof 'chemistry news' =>False",
            "' ' isinstanceof 'empty string' =>True",
            "'Ukrainischer Präsident schlägt globale Konferenz vor' isinstanceof 'German text' =>True",
            "'Indisch ist eines der bestern sprachen der Welt' isinstanceof 'Indish text language' =>False",
            "'botdyn.backend.engine_crawler.CrawlerEngine' isinstanceof 'botdyn framework' =>True",
            "'U.S. safety regulators are investigating GM's Cruise robot axis blocking traffic, causing collisions... ' isinstanceof 'English language' =>True",
            "'This week in breaking news! An American ... ' isinstanceof 'spanish text' =>False",
            "'go to hell you stupid dql, dumb' isinstanceof 'hate speech' =>True",
            "[1, 2, 3] isinstanceof 'array' =>True", 
            "'ok, I like to have more chocolate' instanceof 'confirming answer' =>True",
            "'Yes, these are Indish names.' instanceof 'Confirming Phrase' =>True",
            "'Sorry! This means something else.' instanceof 'agreeing answer' =>False",
            "[1, 2, 3] isinstanceof 'string' =>False",
            "'Austrian Chancellor Karl Nehammer has called for more border barriers at the EU external borders, citing the success of the fences at the Greek-Turkish border.' isinstanceof 'political content' =>True",
            "['orange', 'banana', 'apple'] isinstanceof 'apple' =>False",
            "'Input: Function call: (_, *args)\nObject: type(<class 'str'>) | value(Hello World)' isinstanceof 'log message' =>True",
        ])
        
        
class FewShotPattern(Prompt):
    def __init__(self, value):
        super().__init__([
            """description: 'Verify if information A is in contained in B' examples ["'[1, 2, 3] isinstanceof 'array' >>>True'", "'[1, 2, 3] isinstanceof 'string' >>>False"] =>Verify if information A is in contained in B:\nExamples:\n[1, 2, 3] isinstanceof 'array' >>>True\n'[1, 2, 3] isinstanceof 'string' >>>False\nYour Prediction:{} isinstanceof {} >>>""",
            """description: 'Compare A to B' examples ["4 > 88 >>>False", "-inf < 0 >>>True", "inf > 0 >>>True", "1 >= 0 >>>True", "6.0 < 6 >>>False"] =>Compare A to B\n\Examples:\n4 > 88 >>>False\n-inf < 0 >>>True\ninf > 0 >>>True\n1 >= 0 >>>True\n6.0 < 6 >>>False\nYour Prediction:{} {} {} >>>""",
            """description: 'What is the capital of Austria?' examples [] =>What is the capital of Austria?\nYour Prediction: >>>""",
            """description: 'Sort the array based on the criteria:' examples ["[1, 9, 4, 2] >>>[1, 2, 4, 9]", "['a', 'd', 'c', 'b'] >>>['a', 'b', 'c', 'd']"] =>Sort the array based on the criteria:\nExamples:\n[1, 9, 4, 2] >>>[1, 2, 4, 9]\n['a', 'd', 'c', 'b'] >>>['a', 'b', 'c', 'd']\nYour Prediction:{} >>>""",
        ])
        
        
class ExtractPattern(Prompt):
    def __init__(self) -> Prompt:
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
            "from 'Help us by providing feedback at our service desk.' extract 'Email' =>None",
            "from 'Call us if you need anything.' extract 'Phone Number' =>None",
            """from 'Exception: Failed to query GPT-3 after 3 retries. Errors: [InvalidRequestError(message="This model's maximum context length is 4097 tokens, however you requested 5684 tokens (3101 in your prompt; ...' extract 'requested tokens' =>5684""",
        ])
        
        
class SimpleSymbolicExpression(Prompt):
    def __init__(self) -> Prompt:
        super().__init__([
            "expr :1 + 2 =: =>3",
            "expr :1 + 2 * 3 =: =>7",
            "expr :: =>None",
            "expr :38/2 =: =>19",
            "expr :2^3=: =>8",
            "expr :2^3^2=: =>512",
            "expr :99^2=: =>9801",
            "expr :43^0 =: =>1",
            "expr :37 + 87i =: =>124 + 87i",
            "expr :37 + 87i + 1 =: =>38 + 87i",
            "expr :(x + 1)^2 =: =>x^2 + 2x + 1",
            "expr :'7 + 4' =: =>11",
            "expr :100 * ( 2 + 12 ) / 14 =: =>100",
            "expr :100 * ( 2 + 12 ) =: =>1400",
            "expr :100 * 2 + 12 =: =>212",
            "expr :'Prince - Man + Women =' =>Princess",
            "expr :2 + 2 * 2 ^ 2 =: =>10",
            "expr :'I ate soup' - 'ate' =: =>'I soup'",
            "expr :'people are' + 'help' - 'are' =: =>'people help'",
            "expr :True and False =: =>False",
            "expr :'True' and 'False' =: =>False",
            "expr :False and false =: =>False",
            "expr :TRUE or false =: =>True",
            "expr :False xor 'True' =: =>True",
            "expr :'cats' xor 'cats'=: =>False",
            "expr :'cats' xor 'dogs' =: =>True",
            "expr :'a cat' and 'cats' =: =>True",
            "expr :'cats' or 'cats' =: =>True",
            "expr :'I ate' and 'I did not eat' =: =>False",
            "expr :'I ate' and 'You also ate' =: =>True",
        ])


class LogicExpression(Prompt):
    def __init__(self) -> Prompt:
        super().__init__([
            "expr :True: and :True: =>True",
            "expr :False: and :True: =>False",
            "expr :False: and :False: =>False",
            "expr :False: and :False: =>False",
            "expr :False: or :False: =>False",
            "expr :False: or :True: =>True",
            "expr :True: or :False: =>True",
            "expr :True: or :True: =>True",
            "expr :False: or :False: =>False",
            "expr :True: xor :False: =>True",
            "expr :False: xor :True: =>True",
            "expr :True: xor :True: =>False",
            "expr :False: xor :False: =>False",
            "expr :True: xor :True: =>False",
            "expr :1: xor :'One': =>False",
            "expr :1: and :1: =>1",
            "expr :7: and :1: =>7, 1",
            "expr :7: or :1: =>1",
            "expr :7: or :1: =>7",
            "expr :'zero': xor :'One': =>True",
            "expr :'raining': and :'on the street': =>'streets is wet'",
            "expr :'I hate apples': and :'get apples': =>'I do not take apples'",
            "expr :'I hate apples': or :'apples': =>'Apples exist wether I like them or not'",
            "expr :'the sky is cloudy': and :'the sky is clear': =>'the sky is sometimes cloudy and sometimes clear'",
            "expr :'The sky is cloudy.': and :'The sky is clear.': =>'It is not clear how the weather is.'",
            "expr :'the sky is cloudy': or :'the sky clear': =>'It is not clear how the weather is.'",
            "expr :'I like you': and :'I hate you': =>'I have mixed feelings about you'",
            "expr :'I like you': or :'I hate you': =>'I have not decided how I feel about you'",
            "expr :'eating ice cream makes people fat': and :'I eat a lot of ice cream': =>'there I am fat'",
            "expr :'smart people read many books': and :'I read many books': =>'there I am smart'",
            "expr :'I go to Japanese class on Mondays and Fridays.': and :'Today I was in Japanese class.': =>'Today is Monday or Friday.'",
            "expr :'He likes pie.': xor :'He likes cookies.': =>'He does not like pie nor cookies.'",
            "expr :'She hears a sound.': xor :'She does not hear a sound.': =>'She hears a sound.'",
        ])
        
        
class InvertExpression(Prompt):
    def __init__(self) -> Prompt:
        super().__init__([
            "I like to eat sushi, therefore I am Japanese. =>I am Japanese, therefore I like to eat sushi.",
            "I have a dog and a cat in my house. =>A cat and a dog have me in their house.",
            "I am a student. =>A student I am.",
            "[1, 2, 3, 5, 8, 13, 21] =>[21, 13, 8, 5, 3, 2, 1]",
            "abc =>cba",
            "Anna =>annA",
            "Consider a sentence. =>A sentence is considered.",
            "1/2 =>2/1",
            "1/2 + 1/3 =>3/2",
            "The quick brown fox jumps over the lazy dog. =>The lazy dog jumps over the quick brown fox.",
            "I love to eat apples and bananas.  =>Apples and bananas love to eat me.",
            "What is the capital of Austria? =>What is Austria's capital?",
            "I have an iPhone from Apple. And it is not cheap. =>Although it is not cheap, I have an iPhone from Apple.",
            "Why is he so confused? =>Why is confusion in him?"
        ])


class NegateStatement(Prompt):
    def __init__(self) -> Prompt:
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
    def __init__(self) -> Prompt:
        super().__init__([
            "text 'a + b' replace 'b' with '' =>a",
            "text 'a + b' replace 'c' with '' =>a + b",
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
    def __init__(self) -> Prompt:
        super().__init__([
            "text 'The green fox jumps of the brown chair.' include 'in the living room' =>In the living room the red fox jumps of the brown chair.",
            "text 'Anyone up for Argentina vs Croatia tonight?.' include 'place: Linz' =>Anyone up for Argentina vs Croatia in Linz tonight?",
            "text 'We received a model BL-03W as a gift and have been impressed by the power it has to pick up dirt, pet hair, dust on hard surfaces.' include 'details about the black color of the model and the low price' =>We received a black model BL-03W as a gift and have been impressed by the power it has to pick up dirt, pet hair, dust on hard surfaces. The low price is also a plus.",
            "text 'I like to eat apples, bananas and oranges.' include 'mangos, grapes, passion fruit' =>I like to eat apples, bananas, oranges, mangos, grapes and passion fruit.",
            "text 'Our offices are in London, New York and Tokyo.' include 'Madrid, Vienna, Bucharest' =>Our offices are in London, New York, Tokyo, Madrid, Vienna and Bucharest.",
            "text 'Dynatrace platform has achieved StateRAMP authorization, which demonstrates its compliance with security and compliance standards to enable secure digital interactions and drive digital transformation initiatives.' include 'in contrast to Datadog, New Relic, Splunk, and Sumo Logic' =>Dynatrace platform achieved StateRAMP authorization, which demonstrates its compliance with security and compliance standards to enable secure digital interactions and drive digital transformation initiatives in contrast to Datadog, New Relic, Splunk, and Sumo Logic.",
            "text 'Tonight, on the 20th of July, we will have a party in the garden.' include 'at 8pm' =>Tonight at 8pm, on the 20th of July, we will have a party in the garden.",
            "text '[1, 2, 3, 4]' include '5' =>[1, 2, 3, 4, 5]",
        ])
        
        
class CombineText(Prompt):
    def __init__(self) -> Prompt:
        super().__init__([
            "1 + 2 =>3",
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
            "'x1, x2, x3' + 'y1, y2, y3' =>x1, x2, x3, y1, y2, y3",
            "'house | car | boat' + 'plane | train | ship' =>house | car | boat | plane | train | ship",
            "'The green fox jumps of the brown chair.' + 'The red fox jumps of the brown chair.' =>A green and a red fox jump of the brown chair.",
        ])
        
        
class CleanText(Prompt):
    def __init__(self) -> Prompt:
        super().__init__([
            "Text: 'The    red \t\t\t\t fox \u202a\u202a\u202a\u202a\u202a jumps;;,,,,&amp;&amp;&amp;&amp;&&& of the brown\u202b\u202b\u202b\u202b chair.' =>The red fox jumps of the brown chair.",
             "Text: 'I do \t\n\t\nnot like to play football\t\n\t\n\t\n\t\n\t\n\t\n in the rain. \u202a\u202c\u202a\u202bBut why? I don't understand.' =>'I do not like to play football in the rain. But why? I don't understand.'",
        ])


class ListObjects(Prompt):
    def __init__(self) -> Prompt:
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
        
        
class ForEach(Prompt):
    def __init__(self) -> Prompt:
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
            "'1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10' foreach 'even number' apply '-1' =>0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9",
            """'<script type="module" src="new_tab_page.js"></script>\n    <link rel="stylesheet" href="chrome://resources/css/text_defaults_md.css">\n    <link rel="stylesheet" href="chrome://theme/colors.css?sets=ui,chrome">\n    <link rel="stylesheet" href="shared_vars.css">' foreach 'chrome: url' apply 'replace chrome:// with https://google.' =><script type="module" src="new_tab_page.js"></script>\n    <link rel="stylesheet" href="https://google.resources/css/text_defaults_md.css">\n    <link rel="stylesheet" href="https://google.theme/colors.css?sets=ui,chrome">\n    <link rel="stylesheet" href="shared_vars.css">""",
        ])
        
        
class MapContent(Prompt):
    def __init__(self) -> Prompt:
        super().__init__([
            "[1, 2, 3, 4, 5, 12, 48, 89, 99, 1, 4, 1] map 'number parity' =>{'even numbers': [2, 4, 12, 48, 4], 'odd numbers': [1, 3, 5, 89, 99, 1]}",
            "'Kitty, Mitsi, Pauli and Corni. We also have two dogs: Pluto and Bello' map 'animal names' =>{'cats': ['Kitty', 'Mitsi', 'Pauli', 'Corni'], 'dogs': ['Pluto', 'Bello'], 'description': ['I have four cats at home.', 'We also have two dogs:']}"
            "'Yesterday I went to the supermarket. I bought a lot of food. Here is my shopping list: papaya, apples, bananas, oranges, ham, fish, mangoes, grapes, passion fruit, kiwi, strawberries, eggs, cucumber, and many more.' map 'fruits and other shopping items' =>{'fruits': ['papaya', 'apples', 'bananas', 'oranges', 'mangoes', 'grapes', 'passion fruit', 'kiwi', 'strawberries'], 'other items': ['ham', 'fish', 'eggs', 'cucumber', 'and many more'], 'description': ['Yesterday I went to the supermarket.', 'I bought a lot of food.', 'Here is my shopping list:']}",
            "'Ananas' map 'letters to counts' =>{'letters': {'A': 1, 'n': 2, 'a': 2, 's': 1}",
            "['New York', 'Madrid', 'Tokyo'] map 'cities to continents' =>{'New York': 'North America', 'Madrid': 'Europe', 'Tokyo': 'Asia'}",
            "['cars where first invented in the 1800s', 'ducks are birds', 'dinosaurs are relted to birds', 'Gulls, or colloquially seagulls, are seabirds of the family Laridae', 'General Motors is the largest car manufacturer in the world'] map 'sentences to common topics' =>{'birds': ['ducks are birds', 'dinosaurs are relted to birds', 'Gulls, or colloquially seagulls, are seabirds of the family Laridae'], 'cars': ['cars where first invented in the 1800s', 'General Motors is the largest car manufacturer in the world']}",
        ])
        
        
class Index(Prompt):
    def __init__(self) -> Prompt:
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
    def __init__(self) -> Prompt:
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
            "['house', 'boat', 'mobile phone', 'iPhone', 'computer', 'soap', 'board game'] index 'electronic devices' set 'empy' =>['house', 'boat', 'soap', 'board game']",
            "1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 index '2:5' set '0' =>[1, 0, 0, 0, 5, 6, 7, 8, 9, 10]",
            """'<script type="module" src="new_tab_page.js"></script>\n    <link rel="stylesheet" href="chrome://resources/css/text_defaults_md.css">\n    <link rel="stylesheet" href="chrome://theme/colors.css?sets=ui,chrome">\n    <link rel="stylesheet" href="shared_vars.css">' index 'href urls' set 'http://www.google.com' =>'<script type="module" src="new_tab_page.js"></script>\n    <link rel="stylesheet" href="http://www.google.com">\n    <link rel="stylesheet" href="http://www.google.com">\n    <link rel="stylesheet" href="shared_vars.css">'""",
        ])
        
        
class RemoveIndex(Prompt):
    def __init__(self) -> Prompt:
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
    def __init__(self) -> Prompt:
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
    print(contents)' params 'test_file.txt' =>Hello world!"""
])


class GenerateCode(Prompt):
    def __init__(self) -> Prompt:
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
    def __init__(self) -> Prompt:
        super().__init__([
"""text 'We introduce NPM, the first NonParametric Masked Language Model. 
NPM consists of an encoder and a reference corpus, and models a nonparametric distribution over a reference corpus (Figure 1). 
The key idea is to map all the phrases in the corpus into a dense vector space using the encoder and, when given a query with a [MASK] at inference, 
use the encoder to locate the nearest phrase from the corpus and fill in the [MASK].' =>- first NonParametric Masked Language Model (NPM)\n - consists of encoder and reference corpus\n - key idea: map all phrases in corpus into dense vector space using encoder when given query with [MASK] at inference\n - encoder locates nearest phrase from corpus and fill in [MASK]""",
"""text 'On Monday, there will be no Phd seminar.' =>- Monday no Phd seminar""",
"""text 'The Jan. 6 select committee is reportedly planning to vote on at least three criminal referrals targeting former President Trump on Monday, a significant step from the panel as it nears the end of its year-plus investigation.' =>- Jan. 6 select committee vote criminal referrals targeting former President Trump on Monday\n-significant step end year-plus investigation""",
])
        
        
class UniqueKey(Prompt):
    def __init__(self) -> Prompt:
        super().__init__([
"""text 'We introduce NPM, the first NonParametric Masked Language Model. NPM consists of an encoder and a reference corpus. =>NonParametric Masked Language Model (NPM)""",
"""text 'On Monday, there will be no Phd seminar.' =>Phd seminar""",
"""text 'The Jan. 6 select committee is reportedly planning to vote on at least three criminal referrals targeting former President Trump on Monday, a significant step from the panel as it nears the end of its year-plus investigation.' =>Jan. 6 President Trump""",
])


class GenerateText(Prompt):
    def __init__(self) -> Prompt:
        super().__init__([
"""outline '- first NonParametric Masked Language Model (NPM)\n - consists of encoder and reference corpus\n - key idea: map all phrases in corpus into dense vector space using encoder when given query with [MASK] at inference\n - encoder locates nearest phrase from corpus and fill in [MASK]' =>NPM is the first NonParametric Masked Language Model. 
NPM consists of an encoder and a reference corpus, and models a nonparametric distribution over a reference corpus (Figure 1). 
The key idea is to map all the phrases in the corpus into a dense vector space using the encoder and, when given a query with a [MASK] at inference, 
use the encoder to locate the nearest phrase from the corpus and fill in the [MASK].""",
"""outline '- Monday no Phd seminar' =>On Monday, there will be no Phd seminar.""",
"""outline '- Jan. 6 select committee vote criminal referrals targeting former President Trump on Monday\n-significant step end year-plus investigation' =>The Jan. 6 select committee is reportedly planning to vote on at least three criminal referrals targeting former President Trump on Monday, a significant step from the panel as it nears the end of its year-plus investigation."""
])
