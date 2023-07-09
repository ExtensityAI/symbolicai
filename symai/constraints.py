import json

from .exceptions import ConstraintViolationException, InvalidPropertyException
from .symbol import Symbol


class DictFormatConstraint:
    def __init__(self, format=None):
        if isinstance(format, str):
            self.format = json.loads(format)
        elif isinstance(format, dict):
            self.format = format
        else:
            raise InvalidPropertyException(f"Unsupported format type: {type(format)}")

    def __call__(self, input: Symbol):
        input = Symbol(input)
        try:
            gen_dict = json.loads(input.value)
        except json.JSONDecodeError as e:
            raise ConstraintViolationException(f"Invalid JSON: ```json\n{input.value}\n```\n{e}")
        return DictFormatConstraint.check_keys(self.format, gen_dict)

    @staticmethod
    def check_keys(json_format, gen_dict):
        for key, value in json_format.items():
            if not str(key).startswith('{') and not str(key).endswith('}') and \
                key not in gen_dict or not isinstance(gen_dict[key], type(value)):
                raise ConstraintViolationException(f"Key `{key}` not found or type `{type(key)}` mismatch")
            if isinstance(gen_dict[key], dict):
                # on a dictionary, descend recursively
                return DictFormatConstraint.check_keys(value, gen_dict[key])
        return True
