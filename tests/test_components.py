import pytest

from symai import Expression, PrimitiveDisabler, Symbol


def test_disable_primitives():
    sym1 = Symbol("This is a test")
    sym2 = Symbol("This is another test")
    sym3 = Expression("This is a test")

    with PrimitiveDisabler():
        # disable primitives
        assert all([
            sym1.query('Is this a test?') is None,
            sym1.contains('test') is None,
            sym2.query('Is this a test?') is None,
            sym2.contains('test') is None,
            sym3.query('Is this a test?') is None,
            sym3.contains('test') is None
            ])

    # re-enable primitives
    assert all([
        sym1.query('Is this a test?') is not None,
        sym1.contains('test') is not None,
        sym2.query('Is this a test?') is not None,
        sym2.contains('test') is not None,
        sym3.query('Is this a test?') is not None,
        sym3.contains('test') is not None
        ])


if __name__ == "__main__":
    pytest.main()
