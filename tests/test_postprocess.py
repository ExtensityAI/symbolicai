import pytest
from symai.functional import _postprocess_response, ConstraintViolationException, ProbabilisticBooleanMode
from symai.post_processors import PostProcessor
from typing import Any


# Mock classes and functions
class MockArgument:
    def __init__(self):
        self.prop = MockProp()

class MockProp:
    def __init__(self):
        self.constraints = []

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
    rsp = "hello"
    return_constraint = str
    post_processors = None
    argument = MockArgument()
    mode = ProbabilisticBooleanMode.MEDIUM

    result = _postprocess_response(rsp, return_constraint, post_processors, argument, mode)
    assert result == "hello"

def test_postprocess_response_with_post_processor():
    rsp = "hello"
    return_constraint = str
    post_processors = [MockPostProcessor()]
    argument = MockArgument()
    mode = ProbabilisticBooleanMode.MEDIUM

    result = _postprocess_response(rsp, return_constraint, post_processors, argument, mode)
    assert result == "HELLO"

def test_postprocess_response_with_constraint_satisfied():
    rsp = "hello"
    return_constraint = str
    post_processors = None
    argument = MockArgument()
    argument.prop.constraints = [always_true]
    mode = ProbabilisticBooleanMode.MEDIUM

    result = _postprocess_response(rsp, return_constraint, post_processors, argument, mode)
    assert result == "hello"

def test_postprocess_response_with_constraint_violation():
    rsp = "hello"
    return_constraint = str
    post_processors = None
    argument = MockArgument()
    argument.prop.constraints = [always_false]
    mode = ProbabilisticBooleanMode.MEDIUM

    with pytest.raises(ConstraintViolationException):
        _postprocess_response(rsp, return_constraint, post_processors, argument, mode)

def test_postprocess_response_type_casting():
    rsp = "42"
    return_constraint = int
    post_processors = None
    argument = MockArgument()
    mode = ProbabilisticBooleanMode.MEDIUM

    result = _postprocess_response(rsp, return_constraint, post_processors, argument, mode)
    assert result == 42
    assert isinstance(result, int)

def test_postprocess_response_with_multiple_post_processors():
    rsp = "hello"
    return_constraint = str
    post_processors = [MockPostProcessor1(), MockPostProcessor2()]
    argument = MockArgument()
    mode = ProbabilisticBooleanMode.MEDIUM

    result = _postprocess_response(rsp, return_constraint, post_processors, argument, mode)
    assert result == "HELLO"

def test_postprocess_response_with_multiple_constraints():
    rsp = "hello"
    return_constraint = str
    post_processors = None
    argument = MockArgument()
    argument.prop.constraints = [always_true, always_true]
    mode = ProbabilisticBooleanMode.MEDIUM

    result = _postprocess_response(rsp, return_constraint, post_processors, argument, mode)
    assert result == "hello"

def test_postprocess_response_with_mixed_constraints():
    rsp = "hello"
    return_constraint = str
    post_processors = None
    argument = MockArgument()
    argument.prop.constraints = [always_true, always_false]
    mode = ProbabilisticBooleanMode.MEDIUM

    with pytest.raises(ConstraintViolationException):
        _postprocess_response(rsp, return_constraint, post_processors, argument, mode)


def test_postprocess_response_with_list_return_constraint():
    rsp = ["a", "b", "c"]
    return_constraint = list
    post_processors = None
    argument = MockArgument()
    mode = ProbabilisticBooleanMode.MEDIUM

    result = _postprocess_response(rsp, return_constraint, post_processors, argument, mode)
    assert result == ["a", "b", "c"]
    assert isinstance(result, list)

 
 