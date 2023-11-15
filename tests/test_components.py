import unittest

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


if __name__ == '__main__':
    unittest.main()
