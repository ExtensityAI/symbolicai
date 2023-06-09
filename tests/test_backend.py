import unittest

from symai import Expression, Symbol


class TestBackend(unittest.TestCase):
    def test_index(self):
        expr = Expression()
        expr.add(Symbol('Hello World!').zip())
        expr.add(Symbol('I like cookies!').zip())
        res = expr.get(Symbol('hello').embed().value).ast()
        self.assertTrue(res['matches'][0]['metadata']['text'][0] == 'Hello World!')

