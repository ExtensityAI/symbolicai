import pytest

from symai.components import Interface


def test_interface_unknown_name_raises_clear_error():
    with pytest.raises(ValueError, match="No interface named"):
        Interface("definitely_not_a_real_interface")
