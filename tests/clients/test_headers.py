import pytest

from symai.clients._headers import parse_optional_float, parse_optional_int


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("", None), ("invalid", None), ("2.5", 2.5)],
)
def test_parse_optional_float(value, expected):
    assert parse_optional_float(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("", None), ("invalid", None), ("42", 42)],
)
def test_parse_optional_int(value, expected):
    assert parse_optional_int(value) == expected
