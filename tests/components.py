import unittest

from symai import Symbol, Expression, PrimitiveDisabler


class TestComponents(unittest.TestCase):
    def test_disable_primitives(self):
        def debug(sym):
            try:
                print(sym.query('Is this a test?'))
                print(sym.contains('test'))
            except Exception as e:
                print(f"Error: {e}")
            print('---', '\n')

        sym1 = Symbol("This is a test")
        sym2 = Symbol("This is another test")
        sym3 = Expression("This is a test")

        with PrimitiveDisabler():
            # disable primitives
            self.assertTrue(all([
                sym1.query('Is this a test?') is None,
                sym1.contains('test') is None,
                sym2.query('Is this a test?') is None,
                sym2.contains('test') is None,
                sym3.query('Is this a test?') is None,
                sym3.contains('test') is None
                ])
            )

        # re-enable primitives
        self.assertTrue(all([
            sym1.query('Is this a test?') is not None,
            sym1.contains('test') is not None,
            sym2.query('Is this a test?') is not None,
            sym2.contains('test') is not None,
            sym3.query('Is this a test?') is not None,
            sym3.contains('test') is not None
            ])
        )


if __name__ == "__main__":
    unittest.main()
