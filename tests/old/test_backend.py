import unittest

from symai import Expression, Symbol
from symai.components import Template

Expression.command(time_clock=True)


class TestBackend(unittest.TestCase):
    def test_index(self):
        expr = Expression()
        s1   = Symbol('Hello World!').zip()
        s2   = Symbol('I like cookies!').zip()
        expr.add(s1, index_name='defaultindex')
        expr.add(s2, index_name='defaultindex')
        res = expr.get(Symbol('hello').embed().value).ast()
        self.assertTrue('Hello World!' in res, res)

    def test_html_template(self):
        template = Template()
        res = template(Symbol('Create a table with two columns (title, price).', 'data points: Apple, 1.99; Banana, 2.99; Orange, 3.99'))
        self.assertTrue('<table>' in res, res)


if __name__ == '__main__':
    unittest.main()
