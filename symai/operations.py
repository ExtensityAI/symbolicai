import base64
import re
from collections.abc import Sequence

from symai.runtime.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    ImageContent,
    ImageDetail,
    LanguageModelRequest,
    SamplingConfig,
    SystemMessage,
    TextContent,
    UserMessage,
)

_MEDIA_TYPE_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*"
)


def language_request(
    system_prompt: str,
    user_prompt: str,
    *,
    examples: Sequence[str] = (),
    max_tokens: int | None = None,
    stop: Sequence[str] = (),
) -> LanguageModelRequest:
    """Build a normalized text request from explicit prompt parts.

    An empty part is omitted rather than sent as an empty turn. A request with no content
    at all fails validation instead of reaching a provider.
    """
    messages: list[SystemMessage | UserMessage] = []
    system_parts = ((system_prompt,) if system_prompt else ()) + _string_tuple(
        examples,
        "examples",
    )
    if system_parts:
        messages.append(SystemMessage(content=(TextContent(text="\n".join(system_parts)),)))
    if user_prompt:
        messages.append(UserMessage(content=(TextContent(text=user_prompt),)))

    return LanguageModelRequest(
        messages=tuple(messages),
        sampling=SamplingConfig(max_tokens=max_tokens, stop=_string_tuple(stop, "stop")),
    )


def image_request(
    system_prompt: str,
    user_prompt: str,
    *,
    image_url: str,
    detail: ImageDetail | None = None,
    max_tokens: int | None = None,
    stop: Sequence[str] = (),
) -> LanguageModelRequest:
    messages: list[SystemMessage | UserMessage] = []
    if system_prompt:
        messages.append(SystemMessage(content=(TextContent(text=system_prompt),)))
    messages.append(
        UserMessage(
            content=(
                TextContent(text=user_prompt),
                ImageContent(url=image_url, detail=detail),
            )
        )
    )
    return LanguageModelRequest(
        messages=tuple(messages),
        sampling=SamplingConfig(max_tokens=max_tokens, stop=_string_tuple(stop, "stop")),
    )


def data_uri(data: bytes, media_type: str) -> str:
    if _MEDIA_TYPE_PATTERN.fullmatch(media_type) is None:
        msg = "media_type must be a valid type/subtype token"
        raise ValueError(msg)
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def embedding_request(
    inputs: Sequence[str],
    *,
    dimensions: int | None = None,
    user: str | None = None,
) -> EmbeddingRequest:
    return EmbeddingRequest(
        inputs=_string_tuple(inputs, "inputs"),
        dimensions=dimensions,
        user=user,
    )


def parse_embedding_response(response: EmbeddingResponse) -> list[list[float]]:
    indices = tuple(vector.index for vector in response.vectors)
    if len(indices) != len(set(indices)):
        msg = "Embedding response indices must be unique"
        raise ValueError(msg)
    return [
        [float(value) for value in vector.values]
        for vector in sorted(response.vectors, key=lambda vector: vector.index)
    ]


def _string_tuple(values: Sequence[str], field: str) -> tuple[str, ...]:
    if isinstance(values, str):
        msg = f"{field} must be a sequence of strings, not one string"
        raise TypeError(msg)
    return tuple(values)
