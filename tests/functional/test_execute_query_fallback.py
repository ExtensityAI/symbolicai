import pytest

from symai.functional import _execute_query_fallback


class DummyArgument:
    def __init__(self, args=(), kwargs={}, default=None):
        self.args = args
        self.signature_kwargs = kwargs
        self.prop = type('obj', (object,), {'default': default})


def test_successful_execution():
    def test_func(instance, argument, error, stack_trace, x, y):
        return x + y

    argument = DummyArgument((2, 3))
    result = _execute_query_fallback(func=test_func,instance= None, argument=argument, error=None, stack_trace=None)
    assert result['error'] == 5
    assert result['error'] is None
    assert result['stack_trace'] is None

def test_return_none_with_default():
    def test_func(instance, argument, error, stack_trace):
        return None

    argument = DummyArgument(default="default_value")
    result = _execute_query_fallback(test_func, None, argument, "error_obj", "stack_trace_obj")
    assert result['data'] == "default_value"
    assert result['error'] == "error_obj"
    assert result['stack_trace'] == "stack_trace_obj"

def test_return_value_with_error_info():
    def test_func(instance, argument, error, stack_trace):
        return "success_value"  # Function returns non-None value

    error_obj = ValueError("Test error")
    stack_trace = "mock stack trace"
    argument = DummyArgument()
    result = _execute_query_fallback(test_func, None, argument, error_obj, stack_trace)
    assert result['data'] == "success_value"
    assert result['error'] == error_obj
    assert result['stack_trace'] == stack_trace

def test_exception_without_default():
    def test_func(instance, argument, error, stack_trace):
        return None  # Function returns None

    error_obj = ValueError("Test error")
    argument = DummyArgument()  # No default value
    with pytest.raises(ValueError) as excinfo:
        _execute_query_fallback(test_func, None, argument, error_obj, None)
    assert excinfo.value == error_obj

def test_with_instance_and_kwargs():
    def test_func(instance, argument, error, stack_trace, x, y=10):
        return instance.value + x + y

    class DummyInstance:
        value = 5

    argument = DummyArgument((3,), {'y': 7})
    result = _execute_query_fallback(test_func, DummyInstance(), argument, None, None)
    assert result['data'] == 15

def test_different_default_types():
    def test_func(instance, argument, error, stack_trace):
        return None  # Function returns None, so default is used

    # Test with various default types
    for default_value in [42, [1, 2, 3], {"a": 1}]:
        argument = DummyArgument(default=default_value)
        result = _execute_query_fallback(test_func, None, argument, "error", "trace")
        assert result['data'] == default_value
        assert result['error'] == "error"
        assert result['stack_trace'] == "trace"

def test_instance_modification():
    class MutableInstance:
        def __init__(self):
            self.value = 0

    def test_func(instance, argument, error, stack_trace):
        instance.value += 1
        return instance.value

    instance = MutableInstance()
    argument = DummyArgument()
    result = _execute_query_fallback(test_func, instance, argument, None, None)
    assert result['data'] == 1
    assert instance.value == 1

def test_different_exceptions():
    def test_func(instance, argument, error, stack_trace):
        return None

    value_error = ValueError("Value error")
    key_error = KeyError("Key error")
    
    # Without default, the original error is re-raised
    with pytest.raises(ValueError):
        _execute_query_fallback(test_func, None, DummyArgument(), value_error, None)
    with pytest.raises(KeyError):
        _execute_query_fallback(test_func, None, DummyArgument(), key_error, None)

def test_empty_args_kwargs():
    def test_func(instance, argument, error, stack_trace):
        return "Success"

    argument = DummyArgument()
    result = _execute_query_fallback(test_func, None, argument, None, None)
    assert result['data'] == "Success"

def test_many_arguments():
    def test_func(instance, argument, error, stack_trace, *args, **kwargs):
        return len(args) + len(kwargs)

    argument = DummyArgument(range(10), {f'key_{i}': i for i in range(10)})
    result = _execute_query_fallback(test_func, None, argument, None, None)
    assert result['data'] == 20

def test_error_and_stack_trace_passed_to_func():
    def test_func(instance, argument, error, stack_trace):
        # Return the error and stack trace to verify they're passed correctly
        return (error, stack_trace)

    error_obj = Exception("Test error")
    stack_trace = "Test stack trace"
    argument = DummyArgument()
    result = _execute_query_fallback(test_func, None, argument, error_obj, stack_trace)
    assert result['data'] == (error_obj, stack_trace)