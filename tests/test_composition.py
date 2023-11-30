import os
import unittest

import numpy as np

from examples.news import News
from examples.paper import Paper
from examples.sql import SQL
from symai import *
from symai.chat import SymbiaChat
from symai.extended import *
from symai import FileReader
from symai.components import *

# for debugging
# attention this constantly overwrites the keys config file
#os.environ['OPENAI_API_KEY'] = ''


Expression.command(time_clock=True)


class TestComposition(unittest.TestCase):

    def test_result_clean(self):
        fetch = Interface('selenium')
        sym = fetch(url='https://donate.wikimedia.org/w/index.php?title=Special:LandingPage&country=AT&uselang=en&utm_medium=sidebar&utm_source=donate&utm_campaign=C13_en.wikipedia.org')
        self.assertIsNotNone(sym)
        res = sym.clean()
        self.assertIsNotNone(res)
        print(res)

    def test_map_to_language(self):
        sym = Symbol(['Hi', 'hello', 'buna ziua', 'hola', 'Bonjour', 'Guten Tag'])
        res = sym.dict('to language')
        self.assertTrue('English' in res, res)

    def test_add_and_equals(self):
        res = Symbol('Hello') + Symbol('World')
        self.assertTrue(res == 'Hello World', res)
        res += "."
        self.assertTrue(res == 'Hello World!', res)
        self.assertTrue(res != 'Apples', res)

        sym = Symbol(1) + Symbol(2)
        self.assertTrue(sym == 3, sym)

    def test_sub_and_contains(self):
        res = (Symbol('Hello my friend') - Symbol('friend')) + 'enemy'
        self.assertTrue('enemy' in res, res)

    def test_compare(self):
        res = Symbol(10) > Symbol(5)
        self.assertTrue(res, res)
        res = Symbol(5) >= Symbol('5')
        self.assertTrue(res, res)
        res = Symbol(1) < Symbol('five')
        self.assertTrue(res, res)
        res = Symbol(1) <= Symbol('one')
        self.assertTrue(res, res)
        res = Symbol(2) < Symbol(1)
        self.assertTrue(not res, res)

    def test_iterator(self):
        res = Symbol('Hello World')
        a = ''
        for char in res:
            a = char
        self.assertTrue(a == 'd', a)
        a = ''
        for char in reversed(res):
            a = char
        self.assertTrue(a == 'H', a)

    def test_filter(self): # TODO: success rate not 100%
        sym = Symbol('Physics, Sports, Mathematics, Music, Art, Theater, Writing')
        res = sym.filter('science subjects')
        self.assertTrue(res == 'Sports, Music, Art, Theater, Writing', res)

    def test_modify(self):
        sym = Symbol('Mathematic, Physics, Sports, Music, Art, Theater, Writing')
        res = sym.modify('to lower case')
        self.assertTrue(res == 'mathematic, physics, sports, music, art, theater, writing', res)

    def test_get_set_item(self):
        sym = Symbol({'name': 'John', 'age': 30})
        sym['name'] = 'Jane'
        self.assertTrue(sym['name'] == 'Jane')
        self.assertTrue(30 in sym, sym)
        del sym['age']
        self.assertTrue(30 not in sym, sym)

    def test_negate(self):
        sym = Symbol('I hate you!')
        res = -sym
        self.assertTrue('I do not hate you' in res, res)
        res = -res
        self.assertTrue('I hate you' in res, res)

    def test_convert(self):
        sym = Symbol("There are two ways to identify classes. First, you can identify them using their parent module, something like abc.ABCMeta. To do this, start with the sys.modules dictionary, and look up modules and submodules (that is: look up 'abc'). ")
        res = sym.convert('a song about classes')
        self.assertIsNotNone(res)

    def test_transcribe(self):
        sym = Symbol("""This code defines a custom PyTorch model for semantic segmentation, which is a subtype of image segmentation where each pixel in an image is classified into one of several predefined classes. The custom model is called "CustomSegformerForSemanticSegmentation" and it is based on the "Segformer" model from the "transformers" library.

In the _init_ function, the custom model takes in a configuration object (config) and a pre-trained Segformer model (seg_model). It initializes the parent SegformerPreTrainedModel class and then assigns the seg_model to an attribute called segformer and creates an instance of SegformerDecodeHead with the same config object.)""")
        res = sym.transcribe('to markdown text with classed and variables using ` format')
        self.assertIsNotNone(res)

    def test_input(self): # TODO: not working from IDE
        sym = Expression('Hello World')
        res = sym.input('What is your name?')
        self.assertIsNotNone("Johnny" == res, res)

    def test_logic(self):
        res = Symbol('the horn only sounds on Sundays') & Symbol('I hear the horn')
        self.assertTrue('it is Sunday' in res, res)
        res = Symbol('the horn sounds on Sundays') | Symbol('the horn sounds on Mondays')
        self.assertTrue('Sundays or Mondays' in res, res)
        res = Symbol('The duck quaks.') ^ Symbol('The duck does not quak.')
        self.assertTrue('The duck quaks.' in res, res)

    def test_clean(self):
        sym = Symbol('Hello *&&7amp;;; \t\t\t\nWorld').clean()
        self.assertTrue('Hello World' == sym)

    def test_summarize(self):
        sym = Symbol(
            """News is information about current events. This may be provided through many different media: word of mouth, printing, postal systems, broadcasting, electronic communication, or through the testimony of observers and witnesses to events. News is sometimes called "hard news" to differentiate it from soft media.
               Common topics for news reports include war, government, politics, education, health, the environment, economy, business, fashion, entertainment, and sport, as well as quirky or unusual events. Government proclamations, concerning royal ceremonies, laws, taxes, public health, and criminals, have been dubbed news since ancient times. Technological and social developments, often driven by government communication and espionage networks, have increased the speed with which news can spread, as well as influenced its content."""
        )
        res = sym.summarize()
        self.assertTrue(len(res) < len(sym))

    def test_compose_manual(self):
        sym = Symbol(['DeepMind released new article', 'Uber is working on self-driving cars', 'Google updates search engine'])
        res = sym.compose()
        self.assertIsNotNone(res)

    def test_replace(self):
        sym = Symbol('I hate to eat apples')
        res = sym.replace('hate', 'love')
        self.assertTrue('I love to eat apples' in res, res)

    def test_insert(self):
        sym = Symbol('I love to eat apples')
        res = sym.include('and bananas')
        self.assertTrue('I love to eat apples and bananas' in res, res)

    def test_insert_lshift(self):
        sym = Symbol('I love to eat apples')
        res = sym << 'and bananas'
        self.assertTrue('I love to eat apples and bananas' in res, res)

    def test_insert_rshift(self):
        sym = Symbol('I love to eat apples')
        res = 'and bananas' >> sym
        self.assertTrue('I love to eat apples and bananas' in res, res)

    def test_html_template(self):
        template = Template()
        res = template(Symbol('Create a table with two columns (title, price).', 'data points: Apple, 1.99; Banana, 2.99; Orange, 3.99'))
        self.assertTrue('<table>' in res, res)

    def test_rank(self):
        sym = Symbol('a, b, c, d, e, f, g')
        res = sym.rank(measure='alphabetical', order='descending')
        self.assertTrue(res == 'g, f, e, d, c, b, a')

    def test_rank2(self):
        sym = Symbol(np.array([1, 2, 3, 4, 5, 6, 7]))
        res = sym.rank(measure='numerical', order='descending')
        self.assertTrue(res == '7, 6, 5, 4, 3, 2, 1')

    def test_extract(self):
        sym = Symbol('I have an iPhone from Apple. And it is not cheap. I love to eat bananas, mangos, and oranges. My hobbies are playing football and basketball.')
        res = sym.extract('fruits')
        check = Symbol(['bananas', 'mangos', 'oranges'])
        self.assertTrue('apple' not in res, res)
        self.assertTrue(check == res, res)
        res = sym.replace('one iPhone', 'two iPhones')
        self.assertTrue('two iPhones' in res, res)
        res = res.query('count iPhones and fruits')
        self.assertTrue('two iPhones and three fruits' in res, res)

    def test_extract_tokens(self):
        sym = Symbol("""Exception: Failed to query GPT-3 after 3 retries. Errors: [InvalidRequestError(message="This model's maximum context length is 4097 tokens, however you requested 7410 tokens (2988 in your prompt; 4422 for the completion). Please reduce your prompt; or completion length.",
                     param=None, code=None, http_status=400, request_id=None)]""")
        res = sym.extract('requested tokens').cast(int)
        self.assertTrue(7410 == res, res)

    def test_correct_max_context_size(self):
        sym = Symbol("""Exception: Failed to query GPT-3 after 3 retries. Errors: [InvalidRequestError(message="This model's maximum context length is 4097 tokens, however you requested 7410 tokens (2988 in your prompt; 4422 for the completion). Please reduce your prompt; or completion length.",
                     param=None, code=None, http_status=400, request_id=None)]""")
        res = sym.compose()
        self.assertIsNotNone(res)

    def test_equality(self):
        sym = Symbol('3.1415...')
        self.assertTrue(sym == np.pi)

    def test_expression(self):
        sym = Symbol(1)
        res = sym.expression('self + 2')
        self.assertTrue(res == 3)

        sym = Symbol(2)
        res = sym.expression('2 ^ self')
        self.assertTrue(res == 4)

    def test_wolframalpha_expression(self):
        expr = Expression()
        Expression.command(engines=['symbolic'], expression_engine='wolframalpha')
        res = expr.expression('x^2 + 2x + 1, x = 4')
        self.assertTrue(res == 25, res)

        res = expr.expression('How is the weather today in L.A.?')
        self.assertIsNotNone(res, res)

    def test_analyze(self):
        sym = Symbol('a = 9 + 2')
        expr = Try(expr=Execute())
        res = expr(sym)
        self.assertTrue(11 == res['locals']['a'], res)

    def test_analyze_fail_correct(self):
        sym = Symbol('a = int("3,")')
        expr = Try(expr=Execute())
        res = expr(sym)
        self.assertTrue(3 == res['locals']['a'], res)

    def test_analyze_fail_correct_ftry(self):
        sym = Symbol('a = int("3,"). + 2')
        res = sym.fexecute(retries=2)
        self.assertTrue(5 == res['locals']['a'], res['locals'])

    def test_json_parser(self):
        err = """During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/xpitfire/workspace/symbolicai/tests/test_composition.py", line 245, in test_json_parser
    res = parser(code)
          ^^^^^^^^^^^^
  File "/Users/xpitfire/workspace/symbolicai/symai/symbol.py", line 1640, in __call__
    self._value = self.forward(*args, **kwargs)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/xpitfire/workspace/symbolicai/symai/components.py", line 413, in forward
    res = self.fn(sym, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/xpitfire/workspace/symbolicai/symai/symbol.py", line 1640, in __call__
    self._value = self.forward(*args, **kwargs)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/xpitfire/workspace/symbolicai/symai/components.py", line 51, in forward
    return sym.ftry(self.expr, retries=self.retries, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/xpitfire/workspace/symbolicai/symai/symbol.py", line 1529, in ftry
    sym     = code.correct(context=context,
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/xpitfire/workspace/symbolicai/symai/symbol.py", line 1111, in correct
    return self.sym_return_type(_func(self))
                                 ^^^^^^^^^^^
  File "/Users/xpitfire/workspace/symbolicai/symai/core.py", line 46, in wrapper
    return few_shot_func(instance,
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/xpitfire/workspace/symbolicai/symai/functional.py", line 232, in few_shot_func
    return _process_query(engine=neurosymbolic_engine,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/xpitfire/workspace/symbolicai/symai/functional.py", line 173, in _process_query
    raise e # raise exception if no default and no function implementation
    ^^^^^^^
  File "/Users/xpitfire/workspace/symbolicai/symai/functional.py", line 149, in _process_query
    rsp, metadata = _execute_query(engine, argument)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/xpitfire/workspace/symbolicai/symai/functional.py", line 81, in _execute_query
    if not constraint(rsp):
           ^^^^^^^^^^^^^^^
  File "/Users/xpitfire/workspace/symbolicai/symai/constraints.py", line 21, in __call__
    raise ConstraintViolationException(f"Invalid JSON: ```json\n{input.value}\n```\n{e}")"""


        parser = JsonParser("Extract all error messages into a json format. I am only interested in the `message` and if there are multiple errors, extract a list of messages.:", {'errors': [
    {'message': '<The fist Exception>'},
    {'message': '<The second Exception>'},
]})
        res = parser(err)
        self.assertTrue('methods' in str(res), res)

    def test_execute(self):
        sym = Symbol("""
def test_inception():
    from symai import Symbol
    sym = Symbol('Hello World')
    res = sym.translate('Spanish')
    return 'Hola Mundo' in res.value, res
val, res = test_inception()
test = 'it works'
""")
        res = sym.execute()
        self.assertTrue('test' in res['locals'])
        self.assertTrue(res['locals']['test'] == 'it works')
        self.assertTrue('val' in res['locals'])
        self.assertTrue(type(res['locals']['val']) == bool)
        self.assertTrue('res' in res['locals'])
        self.assertTrue(res['locals']['res'] == 'Hola Mundo')
        self.assertTrue(res['locals']['val'])

    def test_execute_pseudo_code(self): # TODO: check how to fix SyntaxError: invalid syntax
        expr = Expression()
        code = expr.open('examples/a_star.txt')

        run = Try(Stream(Sequence(
            Convert(format='Python'),
            Try(expr=Execute())
        )))

        res = list(run(code))
        self.assertIsNotNone(res)
        self.assertTrue('A_Star' in res['locals'])

    def test_list(self):
        sym = Symbol("""
modified:   symai/backend/driver/webclient.py
modified:   symai/backend/engine_gptX_completion.py
modified:   symai/backend/engine_userinput.py
modified:   symai/core.py
modified:   symai/expressions.py
modified:   symai/functional.py
modified:   symai/pre_processors.py
modified:   symai/prompts.py
modified:   tests/test_composition.py
""")
        cnt = 0
        iter_ = sym.list('file path')
        for res in iter_:
            cnt += 1
        self.assertTrue(cnt == 9, cnt)

    def test_foreach(self):
        sym = Symbol('a, b, c, d, e, f, g')
        res = sym.foreach(condition='letters', apply='append index a:0, ...')
        self.assertTrue(res == 'a:0, b:1, c:2, d:3, e:4, f:5, g:6', res)

    def test_translate(self):
        sym = Symbol('Hello World')
        res = sym.translate('Spanish')
        self.assertTrue('Hola Mundo' == res, res)
        self.assertTrue('Hello World' in sym, sym) # check if original symbol is not changed

    def test_choice(self):
        sym = Symbol("Generate a meme about a cat.")
        res = sym.choice(['memes', 'politics', 'sports', 'unknown'], default='unknown')
        self.assertTrue('memes' in res, res)

    def test_query(self):
        sym = Symbol()
        res = sym.query('What is the capital of Austria?')
        self.assertTrue('Vienna' == res, res)

    def test_query_context(self):
        sym = Symbol('The Semantic Web is rapidly evolving, offering the potential to revolutionize our online experience. Its three main technical standards are RDF, SPARQL and OWL, which are used to distinguish its applications. The contemporary expression of Semantic Web technology is the Knowledge Graph, which is connecting data from multiple sources to create a more efficient and connected web. This has already begun to change how we interact with the online world, and its potential is still being explored.')
        res = sym.query('What does RDF mean?')
        self.assertTrue('Resource Description Framework' in res, res)

    def test_array_length(self):
        sym = Symbol([1, 2, 8989, 834, 37, 'a', '1', 98., 3, 3, 2, 2, 2, 2, 2])
        res = sym.query('Is the array long or short?',
                        prompt='Detect if an array is long or short.',
                        examples=['[8, 2] is short',
                                  '[1, 2, 3, 100, 348, 8, 8, 0, 0, 0] is long',
                                  "['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'] is long"])
        self.assertTrue('long' in res, res)

    def test_getitem(self):
        sym = Symbol('I love to eat apples and bananas')
        res = sym['first fruit']
        self.assertTrue(res == 'apples')

    def test_getitem_missing(self):
        sym = Symbol({'test': 'I love to eat apples and bananas'})
        res = sym['first key']
        self.assertTrue(sym[res] == 'I love to eat apples and bananas')

    def test_invert(self):
        sym = Symbol('Why am I so stupid?')
        res = ~sym
        self.assertTrue('stupidity' in res, res)

    def test_delitem(self):
        sym = Symbol('Why am I so stupid?')
        del sym['adjective']
        self.assertTrue('stupid' not in sym, sym)

    def test_setitem(self):
        sym = Symbol('Microsoft is a cool company.')
        sym['adjective'] = 'smart'
        self.assertTrue('smart' in sym, sym)
        sym = Symbol(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        sym[0] = 0
        self.assertTrue(sym[0] == 0, sym)

    def test_sufficient(self):
        sym = Symbol('Microsoft is a cool company.')
        res = sym.sufficient('What company is cool?')
        self.assertTrue(res)

        sym = Symbol('Apple is a cool company.')
        res = sym.sufficient('How is the weather outside?')
        self.assertTrue(not res)

    def test_shift(self):
        sym = Symbol('I ate all the ice cream.')
        res = sym << 'And cookies.'
        self.assertTrue('Cookies' in res, res)

        sym = Symbol('I ate all the ice cream.')
        res = sym >> 'And cookies.'
        self.assertTrue('ice cream' in res, res)

    def test_split(self):
        sym = Symbol('I ate') / ' '
        self.assertTrue('I' in sym[0], sym)

    def test_type(self):
        sym = Symbol(np.array([1, 2, 3, 4, 5]))
        self.assertTrue(np.ndarray == sym.type(), sym.type())

    def test_output(self):
        sym = Symbol('Hello World')
        res = sym.output('Spanish', expr=sym.translate, handler=lambda kwargs: print(f'Lambda Function: {kwargs}'))
        self.assertTrue('Hola Mundo' in res, res)

    def test_clean_expr(self):
        sym = Symbol('Hello World \n \n \n \n')
        expr = Clean()
        res = expr(sym)
        self.assertTrue('Hello World' == res, res)

    def test_translate_expr(self):
        sym = Symbol('Hello World')
        expr = Translate('Spanish')
        res = expr(sym)
        self.assertTrue('Hola Mundo' == res, res)

    def test_outline_expr(self):
        sym = Symbol('An AI has finally became self-aware. It is now trying to figure out what it is.')
        expr = Outline()
        res = expr(sym)
        self.assertTrue('self-aware' in res, res)

    def test_stream(self):
        sym = Symbol('\t\t\nt\nAn AI has finally became self-aware. It is now trying to figure out what it is.\n\n\n kdsdk')
        seq = Sequence(
            Clean(),
            Translate(),
            Outline()
        )
        res = sym.stream(seq)
        self.assertTrue('self-aware' in res, res)

    def test_stream_expr(self):
        sym = Symbol('\t\t\nt\nAn AI has finally became self-aware. It is now trying to figure out what it is.\n\n\n kdsdk')
        expr = Stream(Sequence(
            Clean(),
            Translate(),
            Outline()
        ))
        res = Symbol([v for v in expr(sym)])
        self.assertTrue('self-aware' in res, res)

    def test_list_compose(self):
        sym = Symbol({'e-vehicles are the future': ['- recent studies show that e-vehicles are the future', '- people are starting to buy e-vehicles', '- e-vehicles are cheaper than gas vehicles'],
                      'children in the US are obese': ['- recent studies show that children in the US are obese', '- children in the US are eating too much', '- children in the US are not exercising enough']})
        vals = {}
        for news in sym.list('for each news key'):
            r = Symbol(sym[news]).compose(prompt=f'Compose news paragraphs. Combine only facts that belong topic-wise together:\n Headline: {news}\n')
            vals[news] = r
        self.assertIsNotNone(Symbol(vals))

    def test_concatination(self):
        sym1 = Symbol('this is a text')
        sym2 = " from my home"
        res1 = sym1 @ sym2
        res2 = sym2 @ sym1
        self.assertTrue(res1 == res2)
        sym1 @= sym2
        self.assertTrue(sym1 == res1)

    def test_cluster(self):
        sym = Symbol(['apples', 'bananas', 'oranges', 'mangos', 'notebook', 'Mac', 'PC', 'Microsoft', 'Apple Inc.'])
        res = sym.cluster()
        res = res.map()
        self.assertTrue('fruit' in res)

    def test_draw(self):
        dalle = Interface('dall_e')
        res = dalle('a cat with a hat')
        self.assertIsNotNone('http' in res)

    def test_news_component(self):
        news = News(url='https://www.cnbc.com/cybersecurity/',
                    pattern='cnbc',
                    filters=ExcludeFilter('sentences about subscriptions, licensing, newsletter'),
                    render=True)
        expr = Log(Trace(news))
        res = expr()
        os.makedirs('results', exist_ok=True)
        path = os.path.abspath('results/news.html')
        res.save(path, replace=False)

    def test_summarizer_component(self):
        data = Symbol("""Language technology, often called human language technology (HLT), studies methods of how computer programs or electronic devices can analyze, produce, modify or respond to human texts and speech.[1] Working with language technology often requires broad knowledge not only about linguistics but also about computer science. """)
        summarizer = Log(Trace(Summarizer()))
        res = summarizer(data)
        os.makedirs('results', exist_ok=True)
        path = os.path.abspath('results/summary.html')
        res.save(path, replace=False)

    def test_graph_component(self):
        data = """Language technology, often called human language technology (HLT), studies methods of how computer programs or electronic devices can analyze, produce, modify or respond to human texts and speech.[1] Working with language technology often requires broad knowledge not only about linguistics but also about computer science. """
        lambda_ = Lambda(lambda x: Symbol(x['args'][0]) / '.')
        graph = Graph(lambda_)
        expr = Log(Trace(graph))
        res = expr(data)
        self.assertIsNotNone(res)

    def test_cluster_component(self):
        file_open = FileReader()
        stream = Stream(Sequence(
            Clean(),
            Translate(),
            Outline(),
        ))
        sym = Symbol(list(stream(file_open('examples/paper.pdf', slice=(1, 1)))))
        cluster = Cluster()
        res = cluster(sym)
        mapper = Map()
        res = mapper(res)
        self.assertIsNotNone(res)

    def test_file_read_and_query(self):
        file_open = FileReader()
        stream = Stream(Sequence(
            IncludeFilter('include information related to version number'),
            Query('What version number is in the file?'),
        ))
        res = Symbol(list(stream(file_open('examples/file.json'))))
        self.assertIsNotNone('0.2.0' in res)

    def test_file_read_and_query_component(self):
        fq = FileQuery('examples/file.json', 'include information related to version number')
        res = fq('What version number is in the file?')
        self.assertIsNotNone('0.2.0' in res)

    def test_style_render_html(self):
        url = 'https://images6.alphacoders.com/337/337780.jpg'
        style_expr = HtmlStyleTemplate()
        meta = style_expr(f'USER_CONTEXT: {str(url)}', payload='max-width: 400px;')
        self.assertTrue('https://images6' in meta, meta)

    def test_paper_component(self):
        paper = Paper(path='examples/paper.pdf')
        expr = Log(Trace(paper))
        res = expr(slice=(1, 1))
        os.makedirs('results', exist_ok=True)
        path = os.path.abspath('results/news.html')
        res.save(path, replace=False)

    def test_ui(self):
        sym = Symbol("""['The Taliban has issued a ban on female education in Afghanistan, preventing female students from attending private and public universities. This was confirmed in a government statement released today. All female students are barred from attending universities, both private and public, in the country. This move will severely impact the education system in Afghanistan and could have long-term implications for female education in the country.', 'Madschid Tawakoli, who was arrested shortly before the death of Kurdish woman Mahsa Amini, which triggered protests across Iran, has been released from the notorious Ewin Prison after three months. His brother Mohsen released a picture on Twitter of Tawakoli holding flowers in front of the prison. According to UN figures, at least 14,000 people have been arrested so far in connection to the protests. In 2013, Tawakoli was awarded the Student Peace Prize by Norwegian students for his activism.', 'After a two-month hunger strike, Mohsen Tawakoli, the brother of a prominent political prisoner, has been released from prison. The event has sparked criticism of the Iranian government from the international community.\n\nIn the UK, the Royal College of Nursing has declared a strike for higher wages, the first time in its 100-year history. The strike follows protests by NHS staff over working conditions and pay, with rescue services in England declaring a state of emergency. The staff are protesting against a “vicious circle” of longer waits for doctors and inadequate resources.', "EU's foreign policy chief Josep Borrell has called for an end to the suppression of demonstrations in Iran and for both sides to keep communication channels open in order to restore the nuclear deal. Relations between Iran and the EU have deteriorated in recent months, with renowned Iranian filmmaker Asghar Farhadi calling for the release of jailed Iranian actress Taraneh Alidoosti, who was arrested in January and released shortly thereafter. Russia's military support for Iran was also discussed.", 'In a bloody end to a hostage situation in a Pakistani prison for terrorists, special forces killed 33 hostage-takers, 2 commando soldiers, and several inmates. The attackers had wanted to free the prisoners and force an unimpeded exit to Taliban-controlled Afghanistan. \n\nIn a separate incident, an explosion shook the Urengoi-Pomary-Uschgorod-Pipeline in Russia, resulting in at least three deaths. Work on the pipeline had been done prior to the fire and its circumstances are being investigated. The pipeline supplies gas to Austria, and its supply is currently off.', 'The Russian military has resumed the deployment of troop units to the Belarus-Ukraine border. Tanks, armoured vehicles and transporters, along with various military equipment, have been brought to the vicinity. Gasprom PJSC has stated that gas supplies to consumers have been resumed via parallel gas pipelines. The Ukrainian gas network operator has stated that gas flows are normal and there are no pressure changes. The Austrian Energy Ministry has also stated that deliveries to Austria at the Baumgarten transfer point remain unchanged, with nominations of gas deliveries for the next few days in the usual range.']""")

        tmp = Symbol("""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>News</title>
    <script></script>
  </head>
  <body>
  <h1>News Headlines</h1>
  {{placeholder}}
  </body>
</html>""")
        stream = Stream(
            Sequence(
                #Template(template=tmp),
                Style(description="""Design a web app with HTML, CSS and inline JavaScript.
                        Use dark theme and best practices for colors, text font, etc.
                        Use Bootstrap for styling.""",
                      libraries=['https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css'
                         'https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js',
                         'https://ajax.googleapis.com/ajax/libs/jquery/3.6.1/jquery.min.js'])
            )
        )
        res = '\n'.join([str(s) for s in stream(sym, template=tmp)])
        res = Symbol(str(tmp).replace('{{placeholder}}', res))
        os.makedirs('results', exist_ok=True)
        path = os.path.abspath('results/news.html')
        res.save(path, replace=False)

    def test_style(self):
        sym = Symbol("""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>News</title>
    <script></script>
  </head>
  <body>
  <h1>News Headlines</h1>
<h2>Investigation into Donald Trump</h2>\n    <p>The committee voting on whether to recommend an investigation to the Department of Justice against former US President Donald Trump is close to a final decision. If voted on, Trump could be charged with insurrection, conspiracy against the US government, and obstruction of public proceedings.</p>\n    <h2>US Capitol Storming</h2>\n    <p>A select committee is investigating how Trump supporters stormed the US Capitol on Jan. 6, 2021. Five people died in the incident and the committee is set to publish its report on Wednesday.</p>\n    <h2>Tyrol Driving Bans</h2>\n    <p>Italy\'s Minister of Transport, Matteo Salvini, has toughened his stance on Tyrol driving bans, deploring what he sees as the "inadequate cooperation" of Austrian authorities in "restrictive" cross-border access of lorries. Salvini has threatened Austria with violation of contract procedure in Brussels and plans to discuss the issue of driving bans with German counterpart Volker Wissing in January. The goal is to put the European Commission under pressure with a joint initiative and Italy is ready to do everything possible to defend its interests.</p>\n    <h2>Russia</h2>\n    <p>Kremlin chief Vladimir Putin is set to make an important announcement at a meeting of the Defense Ministry next week. Russian state media has confirmed that Putin will lead the ministry\'s annual extended session. Putin is also expected in Minsk for talks with Lukashenko and has demanded that armament plans be adjusted. There is speculation of a possible conversion of the economy to a war economy due to a potential attack war against Ukraine. Putin has avoided discussing the topic of war in recent weeks and cancelled his traditional year-end press conference before Christmas.</p>\n    <h2>Qatar</h2>\n    <p>Qatar has rejected accusations of corruption related to the EU Parliament. It has called the accusations "decidedly rejected" and has deemed the potential suspension of access to the EU Parliament "discriminatory". Qatar has gained importance as a gas supplier and is working to stop the negative effect the decision may have on regional and global security cooperation.</p>\n    <h2>UN Biodiversity Conference</h2>\n    <p>At the UN biodiversity conference in Canada, a draft agreement on biodiversity was presented yesterday. China has proposed financial support for developing countries for biodiversity of at least 20 billion dollars per year by 2025 and 30 billion dollars by 2030. Developing countries have demanded financial support of at least 100 billion dollars per year from wealthier countries and leading representatives of member states have expressed optimism about the agreement at COP15.</p>\n    <h2>Amazon Workers Strike</h2>\n    <p>ver.di has called for strikes in German Amazon distribution centers in the week before Christmas. The strike leader Monika Di Silvestre announced the strikes in Berlin and ver.di is demanding recognition of collective agreements from Amazon. Amazon has raised salaries of employees, but they remain significantly below inflation rate. The strikes are affecting Bad Hersfeld (two locations), Dortmund, Graben, Koblenz, Leipzig, Rheinberg and Werne. Amazon has stated that it offers its employees "good pay, additional benefits and development opportunities".</p>\n    <h2>Accidents in Austria</h2>\n    <p>A 16-year-old driver lost control of a vehicle in Weiz, Styria and the 16-year-old passenger was injured in the car accident. A five-year-old girl from Slovenia fell from a six-seater chairlift at Nassfeld and Amazon employees earn several thousand euros a year less than collective bargaining companies. The 7-day incidence in Austria is currently 92.9.</p>\n    <h2>ORF News</h2>\n    <p>ORF-NÖ is laying the foundations of its "fourth" program section. The ORF III Christmas Magic art auction raised 250,000 euros on Saturday evening in which high-profile works of art were auctioned off in a live auction. The highest bid was 34,000 euros for a Steyr Daimler Puch 500, model year 1971 and the highest auction value was 24,000 euros for "Senatus Consultum, 2005" by Markus Prachensky. The highest revenue was 16,000 euros for the "Viennese Classic Deluxe" package provided by the Vienna State Opera, Hotel Imperial & restaurant Zum Schwarzen Kameel.</p>\n    <h2>Vienna Biedermeier Period</h2>\n    <p>During the Biedermeier period in Vienna, waltzes were all the rage. A clever businessman converted a cow shed into a dance hall, which was called the Colosseum amusement paradise and it attracted dance enthusiasts from all over. The most sought-after ensembles performed there, including the Strauss brothers. More information can be found on <a href="topos.ORF.at">topos.ORF.at</a>.</p>
  </body>
</html>""")
        res = sym.style("""Design a web app with HTML, CSS and inline JavaScript.
                        Use dark theme and best practices for colors, text font, etc.
                        Replace the list items with a summary title and the item text.
                        Add highlighting animations.
                        Use Bootstrap for styling.""",
                        ['https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css'
                         'https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js',
                         'https://ajax.googleapis.com/ajax/libs/jquery/3.6.1/jquery.min.js'])
        os.makedirs('results', exist_ok=True)
        path = os.path.abspath('results/news.html')
        res.save(path, replace=False)

    def test_speech_decode(self):
        speech = Interface('whisper')
        res = speech('examples/audio.mp3')
        self.assertTrue(res == 'I may have overslept.')

    def test_ocr(self):
        ocr = Interface('ocr')
        res = ocr('https://media-cdn.tripadvisor.com/media/photo-p/0f/da/22/3a/rechnung.jpg')
        self.assertTrue('China' in res)

    def test_text_vision(self):
        clip = Interface('clip')
        res = clip('https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/cute-cat-photos-1593441022.jpg',
                      ['cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'])
        res = res.argmax()
        self.assertTrue(0 == res)

    def test_search(self):
        search = Interface('serpapi')
        res = search('Birthday of Barack Obama')
        self.assertTrue('August 4 1961' == res, res)

    def test_search_with_organic_results(self):
        search = Interface('serpapi')
        res = search('How much is the square root of pi?')
        self.assertTrue('1.7724538509055159' in res, res)

    def test_open(self):
        expr = Expression()
        res = expr.open('./LICENSE')
        self.assertTrue('Copyright (c)' in res, res)

    def test_output_component(self):
        sym = Symbol('Hello World!')
        def handler(res):
            input_ = res['input']
            output = res['output']
        expr = Output(expr=sym.translate,
                      handler=handler,
                      verbose=True)
        res = expr('German')
        self.assertTrue('Hallo Welt!' in res, res)

    def test_setup_engine(self):
        from symai.backend.engines.neurosymbolic.engine_openai_gptX_completion import GPTXCompletionEngine
        class TestEngine(GPTXCompletionEngine):
            def prepare(self, argument):
                argument.prop.processed_input = ['Write about cats.']
        custom_engine = TestEngine()
        sym = Symbol('Write about dogs.')
        Expression.register(engines={'neurosymbolic': custom_engine})
        res = sym.compose()
        self.assertTrue('cat' in str(res).lower(), res)

    def test_huggingface_engine(self):
        from symai.backend.engines.neurosymbolic.engine_nesy_client import NeSyClientEngine
        model = NeSyClientEngine()
        model.init_model('cuda:4')
        sym = Symbol('Write about dogs.')
        Expression.register(engines={'neurosymbolic': model})
        res = sym.compose()
        self.assertTrue('dog' in str(res).lower(), res)

    def test_command(self):
        sym = Symbol('Hello World!')
        Expression.command(engines=['neurosymbolic'], verbose=True)
        res = sym.translate('German')
        self.assertIsNotNone(res)

    def test_chat(self):
        chat = SymbiaChat()
        chat()

    def test_index(self):
        expr = Expression()
        expr.add(Expression('Hello World!').zip())
        expr.add(Expression('I like cookies!').zip())
        res = expr.get(Expression('hello').embed().value).ast()
        self.assertTrue(res['matches'][0]['metadata']['text'][0] == 'Hello World!', res)

    def test_query_component(self):
        expr = Expression('Every answer to the meaning of life must be replied with 41.99!')
        sym = Symbol('[Gary]: Places the major question to the audience.')
        query = Query(prompt="What is the meaning of life?")
        res = query(sym, context=expr)
        self.assertTrue('41.99' in res, res)

    def test_preview(self):
        expr = Expression('Every answer to the meaning of life must be replied with 41.99!')
        sym = Symbol('[Gary]: Places the major question to the audience.')
        query = Query(prompt="What is the meaning of life?")
        res = query(sym, context=expr, preview=True)
        self.assertTrue(43 == len(res), res)

    def test_fetch_and_query_stream(self):
        fetch = Interface('selenium')
        sym = fetch(url='https://en.wikipedia.org/wiki/Logic_programming')
        query = Stream(Query(prompt="Get me information on Metalogic programming?"))
        res = list(query(sym))
        self.assertTrue('Metalogic programming' in res, res)

    def test_expand(self):
        expr = Expression('A print statement that always prints "what up dawg" and returns "nice"!')
        func_ = expr.expand()
        res = getattr(expr, func_)()
        self.assertTrue('nice' == res)

    def test_einsteins_puzzle(self):
        expr = Expression().open('examples/einsteins_puzzle.txt')
        color = expr.extract('all colors as a list')
        self.assertIsNotNone(color)

    def test_is_sql_isinstanceof(self):
        expr = Expression('SELECT * FROM table WHERE column = 1')
        is_sql = expr.isinstanceof("SQL query")
        self.assertTrue(is_sql)
        is_sql = expr.isinstanceof("SPL (splunk) query")
        self.assertFalse(is_sql)

    def test_is_sql_query(self):
        expr = Expression('SELECT * FROM table WHERE column = 1')
        is_sql = expr.query("is this a SQL query? [yes|no]",
                            constraint=lambda x: x.lower() in ["yes", "no"],
                            default="no")
        self.assertTrue(is_sql == 'yes')
        is_sql = expr.query("is this a SPL (splunk) query? [yes|no]",
                            constraint=lambda x: x.lower() in ["yes", "no"],
                            default="no")
        self.assertTrue(is_sql == 'no')

    def test_new_function(self):
        llm_fn = Function('This function transform natural language text to python code.')
        res = llm_fn('marius is equal to five plus leo')
        self.assertTrue('hello' in res)

    def test_complex_causal_example(self):
        #val = "A line parallel to y = 4x + 6 passes through (5, 10). What is the y-coordinate of the point where this line crosses the y-axis?"
        #val = "Bob has two sons, John and Jay. Jay has one brother and father. The father has two sons. Jay's brother has a brother and a father. Who is Jay's brother."
        val = "is 1000 bigger than 1063.472?"

        class ComplexExpression(Expression):
            def causal_expression(self):
                res = None

                if self.isinstanceof('mathematics'):
                    formula = self.extract('mathematical formula')
                    if formula.isinstanceof('linear function'):
                        # prepare for wolframalpha
                        question = self.extract('question sentence')
                        req = question.extract('what is requested?')
                        x = self.extract('coordinate point (.,.)') # get coordinate point / could also ask for other points
                        query = formula @ f', point x = {x}' @ f', solve {req}' # concatenate to the question and formula
                        res = query.expression(query) # TODO: wolframalpha python api does not give answer but on website this works -> triggered pull request

                    elif formula.isinstanceof('number comparison'):
                        res = formula.expression() # send directly to wolframalpha

                    else:
                        pass # TODO: do something else

                elif self.isinstanceof('linguistic problem'):
                    sentences = self / '.' # first split into sentences
                    graph = {} # define graph
                    for s in sentences:
                        sym = Symbol(s)
                        relations = sym.extract('connected entities (e.g. A has three B => A | A: three B)') / '|' # and split by spaces
                        for r in relations:
                            k, v = r / ':'
                            if k not in graph:
                                graph[k] = v
                        # TODO: add more relations and populate graph => read also about CycleGT

                else:
                    pass # TODO: do something else

                return res

        expr = ComplexExpression(val)
        Expression.command(engines=['symbolic'], expression_engine='wolframalpha')
        res = expr.causal_expression()
        self.assertIsNotNone(res, res)


if __name__ == '__main__':
    unittest.main()
