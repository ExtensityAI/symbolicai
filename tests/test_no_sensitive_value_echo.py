"""No credential, prompt, or provider payload may reach an exception message (FIXPLAN §4).

`authorization_header` is the boundary that vets a credential, but it is not the first
code a bad credential meets: pydantic validates the settings model first, and by default
pydantic quotes the value it rejected into the ValidationError. Anything that reaches an
exception reaches every log that records one, so the model layer has to withhold values
too — these tests pin that, since nothing about `hide_input_in_errors` is self-evident
from reading a model definition.
"""

import io
import logging
import traceback
from collections.abc import Callable

import pytest
from pydantic import SecretStr, ValidationError

from symai.loading import load_runtime
from symai.providers._http.settings import HttpProviderSettings
from symai.runtime.config import EngineConfig, RuntimeConfig
from symai.runtime.models import LanguageModelRequest, TextContent, UserMessage

_SECRET = "sk-DO-NOT-LOG-THIS-KEY"
_PROMPT = "confidential-prompt-body"


def _validation_text(build: Callable[[], object]) -> str:
    """Every channel that records an exception without being asked to include the input.

    This is the guaranteed surface: the message, the arguments, and the traceback — what
    `logger.exception`, an error reporter, and a crash dump all capture. It excludes
    `.errors()`/`.json()`, which take `include_input=True` by default; see
    `test_structured_error_accessors_still_carry_the_input_by_request`.
    """
    with pytest.raises(ValidationError) as raised:
        build()

    error = raised.value
    return "".join(
        (
            str(error),
            str(error.args),
            "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        )
    )


@pytest.mark.parametrize(
    "api_key",
    [
        pytest.param(_SECRET.encode(), id="bytes-key-from-unencoded-subprocess-output"),
        pytest.param([_SECRET], id="list-key"),
        pytest.param({"key": _SECRET}, id="mapping-key"),
        pytest.param(bytearray(_SECRET.encode()), id="bytearray-key"),
    ],
)
def test_a_wrongly_typed_api_key_is_rejected_without_quoting_the_credential(
    api_key: object,
) -> None:
    """A key of the wrong type is rejected by pydantic before the header boundary.

    `open(path, "rb").read()` and `subprocess.check_output(...)` without `.decode()` both
    produce bytes, which is exactly how a real key arrives at this path.
    """
    text = _validation_text(lambda: HttpProviderSettings(api_key=api_key, model="gpt-5.4"))  # pyright: ignore[reportArgumentType]

    assert _SECRET not in text
    assert "api_key" in text


def test_a_wrongly_typed_api_key_does_not_reach_the_log_through_load_runtime() -> None:
    """The end-to-end path a caller actually writes, including `logger.exception`."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger(f"{__name__}.load")
    logger.addHandler(handler)
    logger.propagate = False

    config = RuntimeConfig(
        language_models={
            "chat": EngineConfig(
                implementation="openai:responses",
                settings={"api_key": _SECRET.encode(), "model": "gpt-5.4"},
            )
        },
    )
    try:
        try:
            load_runtime(config)
        except Exception:
            logger.exception("engine load failed")
        else:
            msg = "load_runtime accepted a wrongly typed api_key"
            raise AssertionError(msg)
    finally:
        logger.removeHandler(handler)

    assert _SECRET not in stream.getvalue()


def test_a_rejected_prompt_is_not_quoted_into_the_validation_error() -> None:
    """Passing a prompt where a container is expected must not echo the prompt.

    The rejected value here *is* the prompt, so this is the case where pydantic's default
    input echo would publish it.
    """
    text = _validation_text(lambda: UserMessage(content=_PROMPT))  # pyright: ignore[reportArgumentType]

    assert _PROMPT not in text
    assert "content" in text


def test_a_rejected_request_does_not_quote_the_messages_it_carries() -> None:
    text = _validation_text(
        lambda: LanguageModelRequest(
            messages=(UserMessage(content=(TextContent(text=_PROMPT),)),),
            user=object(),  # pyright: ignore[reportArgumentType]
        )
    )

    assert _PROMPT not in text


def test_a_valid_secret_str_key_still_constructs() -> None:
    settings = HttpProviderSettings(api_key=SecretStr(_SECRET), model="gpt-5.4")

    assert settings.api_key.get_secret_value() == _SECRET
    assert _SECRET not in repr(settings)


def test_structured_error_accessors_still_carry_the_input_by_request() -> None:
    """Pin the exact edge of the guarantee, so nobody assumes it covers more than it does.

    `hide_input_in_errors` scrubs the message, the args, and the traceback — every channel
    that records an exception without being asked for the value. It does not touch
    `.errors()`/`.json()`, which take `include_input=True` by default. Anything that
    serializes a ValidationError for a credential-bearing model must pass
    `include_input=False`.
    """
    with pytest.raises(ValidationError) as raised:
        HttpProviderSettings(api_key=_SECRET.encode(), model="gpt-5.4")  # pyright: ignore[reportArgumentType]

    error = raised.value

    assert _SECRET in str(error.errors())
    assert _SECRET not in str(error.errors(include_input=False))
    assert _SECRET not in str(error)
