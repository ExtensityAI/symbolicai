import unittest

import torch
import symai as ai
from symai.components import *


class TestComponents(unittest.TestCase):
    def test_similarity_classification(self):
        classes = [
            'Bought a new mug.',
            'I wish I had a new mug.',
            'I bought a new mug.'
        ]
        f = SimilarityClassification(classes, in_memory=True)

        x = 'I bought a new mug.'
        y = f(x).value
        self.assertEqual(x, y)

    def test_symbia_capabilities(self):
        f = InContextClassification(ai.prompts.SymbiaCapabilities())

        x = Symbol('How could I solve an elliptical equation?')
        y = f(x)
        self.assertTrue('[WORLD-KNOWLEDGE]' in y.value, y)

        x = Symbol('What is 10 choose 5?')
        y = f(x)
        self.assertTrue('[SYMBOLIC]' in y.value, y)

        x = Symbol('What where who, nevermind!')
        y = f(x)
        self.assertTrue('[DK]' in y.value, y)

        x = Symbol('Give me some flights between Timisoara and Linz.')
        y = f(x)
        self.assertTrue('[SEARCH]' in y.value, y)

        x = Symbol('What does sdfjklas mean?')
        y = f(x)
        self.assertTrue('[DK]' in y.value, y)

        x = Symbol('Find me a recipe for vegan lasagna.')
        y = f(x)
        self.assertTrue('[SEARCH]' in y.value, y)

        x = Symbol('Extract all blog post titles from this website: https://exampleblog.com/')
        y = f(x)
        self.assertTrue('[CRAWLER]' in y.value, y)

        x = Symbol('Transcribe this podcast episode into text: ~/Documents/show.mp3')
        y = f(x)
        self.assertTrue('[SPEECH-TO-TEXT]' in y.value, y)

        x = Symbol('Generate an image of a snowy mountain landscape.')
        y = f(x)
        self.assertTrue('[TEXT-TO-IMAGE]' in y.value, y)

        x = Symbol('Can you read the text from this image of a sign? File: /home/username/Pictures/tokyo.jpg')
        y = f(x)
        self.assertTrue('[OCR]' in y.value, y)

        x = Symbol("Find the author's perspective on artificial intelligence in this article. File: /home/username/Documents/AI_Article.pdf")
        y = f(x)
        self.assertTrue('[RETRIEVAL]' in y.value, y)

        x = Symbol('I have to go now, bye!')
        y = f(x)
        self.assertTrue('[EXIT]' in y.value, y)

        x = Symbol('Can you tell me more about your skills?')
        y = f(x)
        self.assertTrue('[HELP]' in y.value, y)

        x = Symbol('What can you do for me?')
        y = f(x)
        self.assertTrue('[HELP]' in y.value, y)

    def test_indexer(self):
        indexer = Indexer(index_name='dataindex')
        indexer('This is a test!')      # upsert
        index = indexer()
        rsp = index('Is there a test?') # retrieve
        self.assertTrue('confirmation that there is a test' in rsp)


    def test_symbol_typing(self):
        sym = Symbol({1: 'a', 2: 'b', 3: 'c'})
        assert sym.value_type == dict
        sym = Symbol([1, 2, 3])
        assert sym.value_type == list
        sym = Symbol('abc')
        assert sym.value_type == str
        sym = Symbol(1)
        assert sym.value_type == int
        sym = Symbol(1.0)
        assert sym.value_type == float
        sym = Symbol(True)
        assert sym.value_type == bool
        sym = Symbol(Symbol('test'))
        assert sym.value_type == str
        sym.value             == 'test'
        sym = Symbol(Symbol(1))
        assert sym.value_type == int
        sym.value             == 1
        sym = Symbol(Symbol(Symbol(1)))
        assert sym.value_type == int
        assert sym.value      == 1
        sym = Symbol([Symbol(1), Symbol(2), Symbol(3)])
        assert sym.value_type == list
        assert sym.value      == [1, 2, 3]
        sym = Symbol(Symbol([1, 2, 3]))
        assert sym.value_type == list
        assert sym.value      == [1, 2, 3]
        sym = Symbol(Symbol(Symbol([1, 2, 3])))
        assert sym.value_type == list
        assert sym.value      == [1, 2, 3]
        sym = Symbol({Symbol(1): Symbol(Symbol([1, 2, 4])), Symbol(3): Symbol(4)})
        assert sym.value_type == dict
        assert sym.value      == {1: [1, 2, 4], 3: 4}
        arr = np.array(Symbol([Symbol(1), Symbol(3), Symbol(4)]))
        assert np.all(np.array([1, 3, 4]))
        tensor = torch.tensor(Symbol([Symbol(1), Symbol(3), Symbol(4)]))
        assert torch.all(torch.tensor([1, 3, 4]))
        tensor = torch.tensor(Symbol([Symbol(1), Symbol(3), Symbol(4)]), dtype=torch.float32)
        assert torch.all(torch.tensor([1, 3, 4], dtype=torch.float32))


if __name__ == '__main__':
    unittest.main()
