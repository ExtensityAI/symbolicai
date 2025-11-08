import pytest
from box import Box

from symai import Symbol
from symai.core import Argument
from symai.functional import _execute_query_fallback


def test_successful_execution():
    def test_func(instance, error, stack_trace, x, y):
        return x + y

    argument = Argument(
        args=(),
        signature_kwargs={
            'x': 3,
            'y': 4
        },
        decorator_kwargs={}
    )
    instance = None
    result = _execute_query_fallback(func=test_func, instance=instance, argument=argument, error=None, stack_trace=None)
    assert result['data'] == 7
    assert result['error'] is None
    assert result['stack_trace'] is None

def test_return_none_with_default():
    def test_func(instance=None, argument=None, error=None, stack_trace=None):
        return None

    # Create an Argument object with the proper structure
    argument = Argument(
        args=(),
        signature_kwargs={},
        decorator_kwargs={'default': "default_value"}
    )

    result = _execute_query_fallback(test_func, None, argument, "error_obj", "stack_trace_obj")

    assert result['data'] == "default_value"
    assert result['error'] == "error_obj"
    assert result['stack_trace'] == "stack_trace_obj"

def test_return_value_with_error_info():
    def test_func(instance, error, stack_trace):
        return "success_value"  # Function returns non-None value

    error_obj = ValueError("Test error")
    stack_trace = "mock stack trace"

    # Create a proper Argument object
    argument = Argument(
        args=(),
        signature_kwargs={},
        decorator_kwargs={}
    )

    result = _execute_query_fallback(test_func, None, argument, error_obj, stack_trace)

    assert result['data'] == "success_value"
    assert result['error'] == error_obj
    assert result['stack_trace'] == stack_trace

def test_exception_without_default():
    def test_func(instance, error, stack_trace):
        return None  # Function returns None

    error_obj = ValueError("Test error")

    argument = Argument(
        args=(),
        signature_kwargs={},
        decorator_kwargs={}  # No default value specified
    )

    with pytest.raises(ValueError) as excinfo:
        _execute_query_fallback(test_func, None, argument, error_obj, None)
    assert excinfo.value == error_obj

def test_with_instance_and_kwargs():
    def test_func(instance, error, stack_trace, x, y):
        return instance.value + x + y

    class DummyInstance:
        value = 5

    argument = Argument(
        args=(),  # x=3 as positional argument
        signature_kwargs={'y': 7, 'x':3},  # Override default y value
        decorator_kwargs={}
    )

    result = _execute_query_fallback(test_func, DummyInstance(), argument, None, None)
    assert result['data'] == 15  # 5 (instance.value) + 3 (x) + 7 (y) = 15

def test_different_default_types():
    def test_func(instance, error, stack_trace):
        return None  # Function returns None, so default is used

    # Test with various default types
    for default_value in [42, [1, 2, 3], {"a": 1}]:
        # Create a proper Argument object with default value
        argument = Argument(
            args=(),
            signature_kwargs={},
            decorator_kwargs={'default': default_value}
        )

        result = _execute_query_fallback(test_func, None, argument, "error", "trace")

        assert result['data'] == default_value
        assert result['error'] == "error"
        assert result['stack_trace'] == "trace"


def test_empty_args_kwargs():
    def test_func(instance, error, stack_trace):
        return "Success"

    argument = Argument(
        args=(),
        signature_kwargs={},
        decorator_kwargs={}
    )

    result = _execute_query_fallback(test_func, None, argument, None, None)
    assert result['data'] == "Success"


def test_symbol_query_with_preview_true():
    original_symbol_text = "This is a test Symbol for preview functionality."
    query_context = "What is the nature of this symbol?"

    sym = Symbol(original_symbol_text)
    preview_symbol = sym.query(query_context, preview=True)

    assert isinstance(preview_symbol, Symbol), \
        "The result of a preview query should still be a Symbol."

    preview_value = preview_symbol.value
    assert isinstance(preview_value, Box), \
        f"The .value of the preview Symbol should be a Box, got {type(preview_value)} instead."
    assert 'args' in preview_value, "Preview Box should contain 'args'"
    assert 'signature_kwargs' in preview_value, "Preview Box should contain 'signature_kwargs'"
    assert 'decorator_kwargs' in preview_value, "Preview Box should contain 'decorator_kwargs'"
    assert 'kwargs' in preview_value, "Preview Box should contain 'kwargs'"
    assert 'prop' in preview_value, "Preview Box should contain 'prop' (metadata)"

    decorator_kwargs = preview_value['decorator_kwargs']
    assert decorator_kwargs.get('preview') is True, \
        "decorator_kwargs should explicitly state preview is True."
    assert decorator_kwargs.get('context') == query_context, \
        "decorator_kwargs should contain the original query context."
    assert 'prompt' in decorator_kwargs, "decorator_kwargs should contain 'prompt'"
    assert 'examples' in decorator_kwargs, "decorator_kwargs should contain 'examples'"

    # Also check that the original symbol on which query was called remains unchanged
    assert sym.value == original_symbol_text, \
        "The original symbol's value should not be altered by a preview query."
