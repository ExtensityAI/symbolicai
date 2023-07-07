import json

from .symbol import Symbol


class DictFormatConstraint:
    def __init__(self, format=None):
        if isinstance(format, str):
            self.format = json.loads(format)
        elif isinstance(format, dict):
            self.format = format
        else:
            raise Exception(f"Invalid format: {format}")

    def __call__(self, input: Symbol):
        input = Symbol(input)
        try:
            gen_dict = json.loads(input.value)
        except json.JSONDecodeError:
            return False
        return DictFormatConstraint.check_keys(self.format, gen_dict)

    @staticmethod
    def check_keys(json_format, gen_dict):
        for key, value in json_format.items():
            if key not in gen_dict or not isinstance(gen_dict[key], type(value)):
                return False
            if isinstance(gen_dict[key], dict):
                # on a dictionary, descend recursively
                if not DictFormatConstraint.check_keys(value, gen_dict[key]):
                    return False
        return True
