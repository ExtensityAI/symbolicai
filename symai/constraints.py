import json
import logging

from .exceptions import ConstraintViolationException, InvalidPropertyException
from .symbol import Symbol

logger = logging.getLogger(__name__)


class DictFormatConstraint:
    def __init__(self, format=None):
        if isinstance(format, str):
            self.format = json.loads(format)
        elif isinstance(format, dict):
            self.format = format
        else:
            msg = f"Unsupported format type: {type(format)}"
            raise InvalidPropertyException(msg)

    def __call__(self, input: Symbol):
        input_symbol = Symbol(input)
        if input_symbol.value_type is str:
            try:
                gen_dict = json.loads(input_symbol.value)
            except json.JSONDecodeError as e:
                msg = f"Invalid JSON: ```json\n{input_symbol.value}\n```\n{e}"
                raise ConstraintViolationException(msg) from e
            return DictFormatConstraint.check_keys(self.format, gen_dict)
        if input_symbol.value_type is dict:
            return DictFormatConstraint.check_keys(self.format, input_symbol.value)
        msg = f"Unsupported input type: {input_symbol.value_type}"
        raise ConstraintViolationException(msg)

    @staticmethod
    def check_keys(json_format, gen_dict):
        for key, value in json_format.items():
            if (
                not str(key).startswith("{") and not str(key).endswith("}") and key not in gen_dict
            ) or not isinstance(gen_dict[key], type(value)):
                msg = f"Key `{key}` not found or type `{type(key)}` mismatch"
                raise ConstraintViolationException(msg)
            if isinstance(gen_dict[key], dict):
                # on a dictionary, descend recursively
                return DictFormatConstraint.check_keys(value, gen_dict[key])
        return True
