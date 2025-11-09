import json

from .exceptions import ConstraintViolationException, InvalidPropertyException
from .symbol import Symbol
from .utils import CustomUserWarning


class DictFormatConstraint:
    def __init__(self, format=None):
        if isinstance(format, str):
            self.format = json.loads(format)
        elif isinstance(format, dict):
            self.format = format
        else:
            CustomUserWarning(f"Unsupported format type: {type(format)}", raise_with=InvalidPropertyException)

    def __call__(self, input: Symbol):
        input = Symbol(input)
        if input.value_type is str:
            try:
                gen_dict = json.loads(input.value)
            except json.JSONDecodeError as e:
                msg = f"Invalid JSON: ```json\n{input.value}\n```\n{e}"
                CustomUserWarning(msg)
                raise ConstraintViolationException(msg)
            return DictFormatConstraint.check_keys(self.format, gen_dict)
        if input.value_type is dict:
            return DictFormatConstraint.check_keys(self.format, input.value)
        CustomUserWarning(f"Unsupported input type: {input.value_type}", raise_with=ConstraintViolationException)

    @staticmethod
    def check_keys(json_format, gen_dict):
        for key, value in json_format.items():
            if (not str(key).startswith('{') and not str(key).endswith('}') and \
                key not in gen_dict) or not isinstance(gen_dict[key], type(value)):
                CustomUserWarning(f"Key `{key}` not found or type `{type(key)}` mismatch", raise_with=ConstraintViolationException)
            if isinstance(gen_dict[key], dict):
                # on a dictionary, descend recursively
                return DictFormatConstraint.check_keys(value, gen_dict[key])
        return True
