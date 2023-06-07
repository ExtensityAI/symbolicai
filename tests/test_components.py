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
        f = SimilarityClassification(classes)

        x = 'I bought a new mug.'
        y = f(x).value
        self.assertEqual(x, y)

    def test_in_context_classification(self):
        f = InContextClassification(ai.prompts.SymbiaCapabilities())

        x = 'How could I solve an elliptical equation?'
        y = f(x)
        self.assertTrue('internal' in y.value)

        x = 'What is 10 choose 5?'
        y = f(x)
        self.assertTrue('symbolic engine' in y.value)

        x = 'What where who, nevermind!'
        y = f(x)
        self.assertTrue('I don\'t understand' in y.value)

        x = 'Give me some flights between Timisoara and Linz.'
        y = f(x)
        self.assertTrue('search' in y.value)

