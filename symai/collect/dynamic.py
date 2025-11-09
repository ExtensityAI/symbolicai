import ast
import re


class DynamicClass:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return str(self.__dict__)

    @staticmethod
    def from_string(s):
        return create_object_from_string(s)


def create_dynamic_class(class_name, **kwargs):
    return type(class_name, (DynamicClass,), kwargs)()


def parse_custom_class_instances(s):
    pattern = r"(\w+)\((.*?)\)"
    if not isinstance(s, str):
        return s
    matches = re.finditer(pattern, s)

    for match in matches:
        class_name = match.group(1)
        class_args = match.group(2)
        try:
            parsed_args = ast.literal_eval(f'{{{class_args}}}')
        except (ValueError, SyntaxError):
            parsed_args = create_object_from_string(class_args)
        class_instance = create_dynamic_class(class_name, **parsed_args)
        s = s.replace(match.group(0), repr(class_instance))

    return s

# TODO: fix to properly parse nested lists and dicts
def create_object_from_string(str_class):
    def updated_attributes_process(str_class):
        # Regular expression to extract key-value pairs
        attr_pattern = r"(\w+)=(\[.*?\]|\{.*?\}|'.*?'|None|\w+)"
        attributes = re.findall(attr_pattern, str_class)

        # Create an instance of the dynamic class with initial attributes
        updated_attributes = []
        # remove string up until 'content='
        content = str_class.split('ChatCompletionMessage(content=')[-1].split(", role=")[0][1:-1]
        updated_attributes.append(('content', content))
        for key, value in attributes:
            attr_key = key
            attr_value = value
            if attr_key.startswith("'") and attr_key.endswith("'"):
                attr_key = attr_key.strip("'")
            if attr_key.startswith('"') and attr_key.endswith('"'):
                attr_key = attr_key.strip('"')
            if attr_value.startswith("'") and attr_value.endswith("'"):
                attr_value = attr_value.strip("'")
            if attr_value.startswith('"') and attr_value.endswith('"'):
                attr_value = attr_value.strip('"')

            if attr_value.startswith('[') and attr_value.endswith(']'):
                parsed_value = parse_value(attr_value)
                attr_value = parsed_value
                dir(attr_value)
                if hasattr(attr_value, '__dict__'):
                    for k in attr_value.__dict__:
                        v = getattr(attr_value, k)
                        if isinstance(v, str):
                            attr_value[k.strip("'")] = v.strip("'")
            elif attr_value.startswith('{') and attr_value.endswith('}'):
                parsed_value = parse_value(attr_value)
                new_value = {}
                for k in parsed_value:
                    v = parsed_value[k]
                    if isinstance(v, str):
                        v = v.strip("'")
                    new_value[k.strip("'")] = v
                attr_value = new_value
            updated_attributes.append((attr_key, attr_value))
        return updated_attributes

    def parse_value(value):
        try:
            value = parse_custom_class_instances(value)
            if not isinstance(value, str):
                return value
            if value.startswith('['):
                value = value[1:-1]
                values = value.split(',')
                return [parse_value(v.strip()) for v in values]
            if value.startswith('{'):
                value = value[1:-1]
                values = value.split(',')
                return {k.strip(): parse_value(v.strip()) for k, v in [v.split(':', 1) for v in values]}
            res = ast.literal_eval(value)
            if isinstance(res, dict):
                return {k: parse_value(v) for k, v in res.items()}
            if isinstance(res, (list, tuple, set)):
                return [parse_value(v) for v in res]
            return res
        except (ValueError, SyntaxError):
            return value

    updated_attributes = updated_attributes_process(str_class)
    return DynamicClass(**{key: parse_value(value) for key, value in updated_attributes})
