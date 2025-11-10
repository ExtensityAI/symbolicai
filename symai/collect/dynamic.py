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


def _strip_quotes(text):
    if not isinstance(text, str):
        return text
    if text.startswith("'") and text.endswith("'"):
        return text.strip("'")
    if text.startswith('"') and text.endswith('"'):
        return text.strip('"')
    return text


def _extract_content(str_class):
    return str_class.split('ChatCompletionMessage(content=')[-1].split(", role=")[0][1:-1]


def _parse_value(value):
    try:
        value = parse_custom_class_instances(value)
        if not isinstance(value, str):
            return value
        if value.startswith('['):
            inner_values = value[1:-1]
            values = inner_values.split(',')
            return [_parse_value(v.strip()) for v in values]
        if value.startswith('{'):
            inner_values = value[1:-1]
            values = inner_values.split(',')
            return {k.strip(): _parse_value(v.strip()) for k, v in [v.split(':', 1) for v in values]}
        result = ast.literal_eval(value)
        if isinstance(result, dict):
            return {k: _parse_value(v) for k, v in result.items()}
        if isinstance(result, (list, tuple, set)):
            return [_parse_value(v) for v in result]
        return result
    except (ValueError, SyntaxError):
        return value


def _process_list_value(raw_value):
    parsed_value = _parse_value(raw_value)
    dir(parsed_value)
    if hasattr(parsed_value, '__dict__'):
        for key in parsed_value.__dict__:
            value = getattr(parsed_value, key)
            if isinstance(value, str):
                parsed_value[key.strip("'")] = value.strip("'")
    return parsed_value


def _process_dict_value(raw_value):
    parsed_value = _parse_value(raw_value)
    new_value = {}
    for key, value in parsed_value.items():
        stripped_value = value.strip("'") if isinstance(value, str) else value
        new_value[key.strip("'")] = stripped_value
    return new_value


def _collect_attributes(str_class):
    attr_pattern = r"(\w+)=(\[.*?\]|\{.*?\}|'.*?'|None|\w+)"
    attributes = re.findall(attr_pattern, str_class)
    updated_attributes = [('content', _extract_content(str_class))]
    for key, raw_value in attributes:
        attr_key = _strip_quotes(key)
        attr_value = _strip_quotes(raw_value)
        if attr_value.startswith('[') and attr_value.endswith(']'):
            attr_value = _process_list_value(attr_value)
        elif attr_value.startswith('{') and attr_value.endswith('}'):
            attr_value = _process_dict_value(attr_value)
        updated_attributes.append((attr_key, attr_value))
    return updated_attributes


# TODO: fix to properly parse nested lists and dicts
def create_object_from_string(str_class):
    updated_attributes = _collect_attributes(str_class)
    return DynamicClass(**{key: _parse_value(value) for key, value in updated_attributes})
