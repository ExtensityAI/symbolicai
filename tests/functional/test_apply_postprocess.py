from typing import Any

import pytest

from symai.functional import (
    ConstraintViolationException,
    ProbabilisticBooleanMode,
    _apply_postprocessors,
)
from symai.post_processors import PostProcessor


# Mock classes and functions
class MockArgument:
    def __init__(self):
        self.prop = MockProp()


class MockProp:
    def __init__(self):
        self.constraints = []
        self.outputs = None
        self.metadata = None
        self.raw_output = None
        self.preview = False


class MockPostProcessor(PostProcessor):
    def __call__(self, rsp: Any, argument: Any) -> Any:
        return rsp.upper()


class MockPostProcessor1(PostProcessor):
    def __call__(self, rsp: Any, argument: Any) -> Any:
        return rsp.upper()


class MockPostProcessor2(PostProcessor):
    def __call__(self, rsp: Any, argument: Any) -> Any:
        return rsp + " world"


def always_true(x):
    return True


def always_false(x):
    return False

# Test cases
def test_postprocess_response_basic():
    rsp = ["hello"]
    return_constraint = str
    post_processors = None
    argument = MockArgument()
    argument.prop.outputs = (rsp, None)
    mode = ProbabilisticBooleanMode.MEDIUM

    result, metadata = _apply_postprocessors(argument.prop.outputs, return_constraint, post_processors, argument, mode)
    assert result == "hello"
    assert metadata is None


def test_postprocess_response_with_post_processor():
    rsp = ["hello"]
    return_constraint = str
    post_processors = [MockPostProcessor()]
    argument = MockArgument()
    argument.prop.outputs = (rsp, None)
    mode = ProbabilisticBooleanMode.MEDIUM

    result, metadata = _apply_postprocessors(argument.prop.outputs, return_constraint, post_processors, argument, mode)
    assert result == "HELLO"
    assert metadata is None


def test_postprocess_response_with_constraint_satisfied():
    rsp = ["hello"]
    return_constraint = str
    post_processors = None
    argument = MockArgument()
    argument.prop.constraints = [always_true]
    argument.prop.outputs = (rsp, None)
    mode = ProbabilisticBooleanMode.MEDIUM

    result, metadata = _apply_postprocessors(argument.prop.outputs, return_constraint, post_processors, argument, mode)
    assert result == "hello"
    assert metadata is None


def test_postprocess_response_with_constraint_violation():
    rsp = ["hello"]
    return_constraint = str
    post_processors = None
    argument = MockArgument()
    argument.prop.constraints = [always_false]
    argument.prop.outputs = (rsp, None)
    mode = ProbabilisticBooleanMode.MEDIUM

    with pytest.raises(ConstraintViolationException):
        _apply_postprocessors(argument.prop.outputs, return_constraint, post_processors, argument, mode)


def test_postprocess_response_type_casting():
    rsp = ["42"]
    return_constraint = int
    post_processors = None
    argument = MockArgument()
    argument.prop.outputs = (rsp, None)
    mode = ProbabilisticBooleanMode.MEDIUM

    result, metadata = _apply_postprocessors(argument.prop.outputs, return_constraint, post_processors, argument, mode)
    assert result == 42
    assert isinstance(result, int)
    assert metadata is None


def test_postprocess_response_with_multiple_post_processors():
    rsp = ["hello"]
    return_constraint = str
    post_processors = [MockPostProcessor1(), MockPostProcessor2()]
    argument = MockArgument()
    argument.prop.outputs = (rsp, None)
    mode = ProbabilisticBooleanMode.MEDIUM

    result, metadata = _apply_postprocessors(argument.prop.outputs, return_constraint, post_processors, argument, mode)
    assert result == "HELLO world"
    assert metadata is None


def test_postprocess_response_with_multiple_constraints():
    rsp = ["hello"]
    return_constraint = str
    post_processors = None
    argument = MockArgument()
    argument.prop.constraints = [always_true, always_true]
    argument.prop.outputs = (rsp, None)
    mode = ProbabilisticBooleanMode.MEDIUM

    result, metadata = _apply_postprocessors(argument.prop.outputs, return_constraint, post_processors, argument, mode)
    assert result == "hello"
    assert metadata is None


def test_postprocess_response_with_mixed_constraints():
    rsp = ["hello"]
    return_constraint = str
    post_processors = None
    argument = MockArgument()
    argument.prop.constraints = [always_true, always_false]
    argument.prop.outputs = (rsp, None)
    mode = ProbabilisticBooleanMode.MEDIUM

    with pytest.raises(ConstraintViolationException):
        _apply_postprocessors(argument.prop.outputs, return_constraint, post_processors, argument, mode)


def test_postprocess_response_with_list_return_constraint():
    rsp = [["a", "b", "c"]]
    return_constraint = list
    post_processors = None
    argument = MockArgument()
    argument.prop.outputs = (rsp, None)
    mode = ProbabilisticBooleanMode.MEDIUM

    result, metadata = _apply_postprocessors(argument.prop.outputs, return_constraint, post_processors, argument, mode)
    assert result == ["a", "b", "c"]
    assert isinstance(result, list)
    assert metadata is None


def test_postprocess_response_with_preview_mode():
    outputs_tuple = (["hello"], {"some_meta": "data"})
    return_constraint = int  # This should be ignored in preview mode
    post_processors = [MockPostProcessor()] # This should be ignored
    argument = MockArgument()
    argument.prop.outputs = outputs_tuple
    argument.prop.preview = True
    argument.prop.constraints = [always_false] # This should be ignored

    outputs_param_to_function = (["raw_input_string"], {"meta": "info"})

    result = _apply_postprocessors(outputs_param_to_function, return_constraint, post_processors, argument, ProbabilisticBooleanMode.MEDIUM)

    # In preview mode, the function should return the `outputs` argument directly.
    assert result == outputs_param_to_function
