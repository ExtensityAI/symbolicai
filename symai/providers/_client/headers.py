from pydantic import SecretStr


def authorization_header(api_key: SecretStr) -> str:
    if not isinstance(api_key, SecretStr):
        raise TypeError

    value = api_key.get_secret_value()
    invalid = not value or value[0].isspace() or value[-1].isspace()
    if not invalid:
        for character in value:
            code_point = ord(character)
            if code_point < 0x20 or code_point == 0x7F:
                invalid = True
                break

    if invalid:
        raise ValueError

    return f"Bearer {value}"


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
