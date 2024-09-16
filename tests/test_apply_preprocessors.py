import pytest
from symai.functional import _apply_preprocessors
from symai.pre_processors import PreProcessor
from types import SimpleNamespace

class MockArgument:
    def __init__(self, raw_input=False, args=None, instance=""):
        self.prop = SimpleNamespace(raw_input=raw_input, instance=instance)
        self.args = args or []

class MockPreProcessor(PreProcessor):
    def __call__(self, argument):
        return f"Processed: {argument.prop.instance}"

@pytest.mark.parametrize("raw_input,args,pre_processors,instance,expected", [
    (False, [], [MockPreProcessor()], "test_instance", "Processed: test_instance"),
    (True, ["arg1", "test_instance"], None, "test_instance", "arg1 test_instance"),
    (False, [], None, "test_instance", "test_instance"),
    (False, ["arg1", "arg2"], None, "test_instance", "test_instance"),
    (False, [], [], "test_instance", "test_instance"),   
])
def test_apply_preprocessors(raw_input, args, pre_processors, instance, expected):
    argument = MockArgument(raw_input=raw_input, args=args)
    result = _apply_preprocessors(argument, instance, pre_processors)
    assert result == expected

def test_apply_preprocessors_multiple():
    class AnotherMockPreProcessor(PreProcessor):
        def __call__(self, argument):
            return f"Also processed: {argument.prop.instance}"

    argument = MockArgument(raw_input=False)
    pre_processors = [MockPreProcessor(), AnotherMockPreProcessor()]
    result = _apply_preprocessors(argument, "test_instance", pre_processors)
    assert result == "Processed: test_instanceAlso processed: test_instance"

def test_apply_preprocessors_none_return():
    class NoneReturnPreProcessor(PreProcessor):
        def __call__(self, argument):
            return None

    argument = MockArgument(raw_input=False)
    pre_processors = [NoneReturnPreProcessor(), MockPreProcessor()]
    result = _apply_preprocessors(argument, "test_instance", pre_processors)
    assert result == "Processed: test_instance"

def test_apply_preprocessors_modifying():
    class ModifyingPreProcessor(PreProcessor):
        def __call__(self, argument):
            argument.prop.instance = "modified_" + argument.prop.instance
            return argument.prop.instance

    argument = MockArgument(raw_input=False, instance="test_instance")
    pre_processors = [ModifyingPreProcessor(), MockPreProcessor()]
    result = _apply_preprocessors(argument, "test_instance", pre_processors)
    assert result == "modified_test_instanceProcessed: modified_test_instance"


def test_apply_preprocessors_exception():
    class ExceptionPreProcessor(PreProcessor):
        def __call__(self, argument):
            raise ValueError("Test exception")

    argument = MockArgument(raw_input=False)
    pre_processors = [ExceptionPreProcessor(), MockPreProcessor()]
    with pytest.raises(ValueError, match="Test exception"):
        _apply_preprocessors(argument, "test_instance", pre_processors)
