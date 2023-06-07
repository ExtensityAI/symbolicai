import unittest

from symai.components import *


class TestComponents(unittest.TestCase):
    def test_similarity_classification(self):
        classes = [
            'Bought a new mug.',
            'I wish I had a new mug.',
            'I bought a new mug.'
        ]
        x = 'I bought a new mug.'

        sim = SimilarityClassification(classes)
        y = sim(x).value

        self.assertEqual(x, y)
