import unittest

from symai import Interface, Strategy
from symai.components import Function


class TestStrategy(unittest.TestCase):
    def test_long_text_strategy(self):
        file = Interface('file')
        data = file('symai/symbol.py')
        strategy = Strategy('ChainOfThought')
        # with Strategy('ChainOfThought') as strategy:
        #     fn = Function('Which method is best used to compose poems from a given symbol?')
        #     res = fn(data)
        res = strategy('Which method is best used to compose poems from a given symbol?' | data)
        self.assertTrue('compose' in res)


if __name__ == '__main__':
    unittest.main()
