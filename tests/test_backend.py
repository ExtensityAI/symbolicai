import unittest

from symai import Expression, Symbol, Template


Expression.command(time_clock=True)


class TestBackend(unittest.TestCase):
    def test_index(self):
        expr = Expression()
        expr.add(Symbol('Hello World!').zip())
        expr.add(Symbol('I like cookies!').zip())
        res = expr.get(Symbol('hello').embed().value).ast()
        self.assertTrue(res['matches'][0]['metadata']['text'][0] == 'Hello World!')

    def test_html_template(self):
        template = Template()
        template(Symbol('Create a table with two columns (title, price).', 'data points: Apple, 1.99; Banana, 2.99; Orange, 3.99'))
        self.assertTrue('<table>' in template, template)

