import pytest
from symai.functional import _limit_number_results
from types import SimpleNamespace

@pytest.fixture
def argument():
    arg = SimpleNamespace()
    arg.prop = SimpleNamespace()
    return arg

class TestLimitNumberResults:
    def test_string_list_limit(self, argument):
        argument.prop.limit = 2
        rsp = ['a', 'b', 'c']
        result = _limit_number_results(rsp, argument, str)
        assert result == 'a\nb'

    def test_list_limit(self, argument):
        argument.prop.limit = 2
        rsp = [1, 2, 3, 4]
        result = _limit_number_results(rsp, argument, list)
        assert result == [1, 2]

    def test_dict_limit(self, argument):
        argument.prop.limit = 2
        rsp = {'a': 1, 'b': 2, 'c': 3}
        result = _limit_number_results(rsp, argument, dict)
        assert result == {'a': 1, 'b': 2}

    def test_set_limit(self, argument):
        argument.prop.limit = 2
        rsp = {1, 2, 3, 4}
        result = _limit_number_results(rsp, argument, set)
        assert len(result) == 2
        assert all(x in {1, 2, 3, 4} for x in result)

    def test_tuple_limit(self, argument):
        argument.prop.limit = 2
        rsp = (1, 2, 3, 4)
        result = _limit_number_results(rsp, argument, tuple)
        assert result == (1, 2)

    def test_no_limit(self, argument):
        argument.prop.limit = None
        rsp = [1, 2, 3, 4]
        result = _limit_number_results(rsp, argument, list)
        assert result == [1, 2, 3, 4]

    def test_limit_greater_than_length(self, argument):
        argument.prop.limit = 10
        rsp = [1, 2, 3]
        result = _limit_number_results(rsp, argument, list)
        assert result == [1, 2, 3]

    def test_non_iterable_input(self, argument):
        argument.prop.limit = 2
        rsp = 42
        result = _limit_number_results(rsp, argument, int)
        assert result == 42

    def test_limit_of_one(self, argument):
        argument.prop.limit = 1
        rsp = [1, 2, 3]
        result = _limit_number_results(rsp, argument, list)
        assert result == [1]

    def test_empty_input(self, argument):
        argument.prop.limit = 2
        rsp = []
        result = _limit_number_results(rsp, argument, list)
        assert result == []

    def test_string_input(self, argument):
        argument.prop.limit = 3
        rsp = "hello"
        result = _limit_number_results(rsp, argument, str)
        assert result == "hello"  

   