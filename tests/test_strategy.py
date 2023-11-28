import unittest

from symai import Interface, Strategy
from symai.components import Function

class TestStrategy(unittest.TestCase):
    def test_long_text_strategy(self):
        file = Interface('file')
        data = file('symai/symbol.py')
        fn = Function('Which method is best used to compose poems from a given symbol?')
        strategy = Strategy('longtext', expr=fn)
        res = strategy(data)
        self.assertTrue('compose' in res)


if __name__ == '__main__':
    unittest.main()
