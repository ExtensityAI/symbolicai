import pytest
import inspect
from symai.functional import _cast_return_type, ProbabilisticBooleanMode, ConstraintViolationException
from typing import Any, Type, Union, List, Dict

@pytest.mark.parametrize("rsp, return_constraint, expected_output", [
    # String tests
    ("test", str, "test"),
    (42, str, "42"),
    
    # Integer tests
    ("42", int, 42),
    (42, int, 42),
    
    # Float tests
    ("3.14", float, 3.14),
    (3.14, float, 3.14),
    
    # List tests
    ("[1, 2, 3]", list, [1, 2, 3]),
    ("['a', 'b', 'c']", list, ['a', 'b', 'c']),
    ([1, 2, 3], list, [1, 2, 3]),
    
    # Tuple tests
    ("(1, 2, 3)", tuple, (1, 2, 3)),
    ("('a', 'b', 'c')", tuple, ('a', 'b', 'c')),
    ((1, 2, 3), tuple, (1, 2, 3)),
    
    # Set tests
    ("{1, 2, 3}", set, {1, 2, 3}),
    ("{'a', 'b', 'c'}", set, {'a', 'b', 'c'}),
    ({1, 2, 3}, set, {1, 2, 3}),
    
    # Dict tests
    ('{"a": 1, "b": 2}', dict, {"a": 1, "b": 2}),
    ({"a": 1, "b": 2}, dict, {"a": 1, "b": 2}),
    
    # Boolean tests
    ("True", bool, True),
    ("False", bool, False),
    ("1", bool, True),
    ("0", bool, False),
    (True, bool, True),
    (False, bool, False),
    
])
def test_cast_return_type(rsp: Any, return_constraint: Type, expected_output: Any):
    result = _cast_return_type(rsp, return_constraint, ProbabilisticBooleanMode.TOLERANT)
    assert result == expected_output
    assert isinstance(result, return_constraint)

def test_cast_return_type_empty():
    result = _cast_return_type("test", inspect._empty, ProbabilisticBooleanMode.TOLERANT)
    assert result == "test"

@pytest.mark.parametrize("invalid_input, return_constraint", [
    ("not a list", list),
    ("not a tuple", tuple),
    ("not a set", set),
    ("not a dict", dict),
])
def test_cast_return_type_invalid_input(invalid_input, return_constraint):
    with pytest.warns(UserWarning):
        result = _cast_return_type(invalid_input, return_constraint, ProbabilisticBooleanMode.TOLERANT)
    assert result == invalid_input

@pytest.mark.parametrize("empty_input, return_constraint, expected", [
    ("", bool, False),
    ([], bool, False),
    ({}, bool, False),
    (set(), bool, False),
])
def test_cast_return_type_empty_bool(empty_input, return_constraint, expected):
    result = _cast_return_type(empty_input, return_constraint, ProbabilisticBooleanMode.TOLERANT)
    assert result == expected

@pytest.mark.parametrize("input_value, mode, expected", [
    ("yes", ProbabilisticBooleanMode.STRICT, False),
    ("true", ProbabilisticBooleanMode.STRICT, True),
    ("yes", ProbabilisticBooleanMode.MEDIUM, True),
    ("1", ProbabilisticBooleanMode.MEDIUM, False),
    ("maybe", ProbabilisticBooleanMode.TOLERANT, False),
    ("no", ProbabilisticBooleanMode.TOLERANT, False),
])
def test_cast_return_type_bool_modes(input_value, mode, expected):
    result = _cast_return_type(input_value, bool, mode)
    assert result == expected

def test_cast_return_type_custom_class():
    class CustomClass:
        def __init__(self, value):
            self.value = value
    
    result = _cast_return_type("42", CustomClass, ProbabilisticBooleanMode.TOLERANT)
    assert isinstance(result, CustomClass)
    assert result.value == "42"

@pytest.mark.parametrize("input_value, return_constraint", [
    (42, str),
    ("42", int),
    (3.14, int),
    ([1, 2, 3], str),
    ("['not', 'a', 'list']", list),
])
def test_cast_return_type_type_conversion(input_value, return_constraint):
    result = _cast_return_type(input_value, return_constraint, ProbabilisticBooleanMode.TOLERANT)
    assert isinstance(result, return_constraint)

def test_cast_return_type_none():
    result = _cast_return_type(None, str, ProbabilisticBooleanMode.TOLERANT)
    assert result == "None"

def test_cast_return_type_invalid_mode():
    with pytest.raises(ValueError):
        _cast_return_type("yes", bool, "INVALID_MODE")

def test_cast_return_type_constraint_violation():
    with pytest.raises(ConstraintViolationException):
        _cast_return_type("not an int", int, ProbabilisticBooleanMode.STRICT)

@pytest.mark.parametrize("input_value, return_constraint, expected", [
        ({'a': [1, 2], 'b': {'c': 3}}, Dict[str, Union[List[int], Dict[str, int]]], {'a': [1, 2], 'b': {'c': 3}}),
        ('🐍🚀', str, '🐍🚀'),   
    ])

def test_cast_return_type_complex(input_value, return_constraint, expected):
    result = _cast_return_type(input_value, return_constraint, ProbabilisticBooleanMode.TOLERANT)
    assert result == expected
    assert isinstance(result, return_constraint)