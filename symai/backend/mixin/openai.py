from ...prompts import CACHE_BREAKPOINT, split_cache_breakpoints, strip_cache_breakpoints
from ...utils import UserMessage


def build_cache_breakpoint_blocks(text: str) -> list[dict]:
    segments = split_cache_breakpoints(text)
    breakpoint_count = len(segments) - 1
    if breakpoint_count > 4:
        msg = "OpenAI supports at most four cache breakpoint writes per request."
        raise ValueError(msg)
    if any(segment == "" for segment in segments[:-1]):
        msg = "OpenAI cache breakpoints must follow non-empty text segments."
        raise ValueError(msg)

    blocks = []
    for index, segment in enumerate(segments):
        if segment == "":
            continue
        block = {"type": "input_text", "text": segment}
        if index < breakpoint_count:
            # TODO: Use the SDK breakpoint type once input content params expose it.
            block["prompt_cache_breakpoint"] = {"mode": "explicit"}
        blocks.append(block)
    return blocks


SUPPORTED_COMPLETION_MODELS = [
    "davinci-002",
]
SUPPORTED_CHAT_MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0613",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-1106-preview",  # @NOTE: probabily obsolete; same price as 'gpt-4-turbo-2024-04-09' but no vision
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini",
    "chatgpt-4o-latest",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-5-chat-latest",
    "gpt-5.1-chat-latest",
    "gpt-5.2-chat-latest",
]
SUPPORTED_REASONING_MODELS = [
    "o3-mini",
    "o4-mini",
    "o1",
    "o3",
    "gpt-5",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.5",
    "gpt-5.5-2026-04-23",
]
SUPPORTED_RESPONSES_REASONING_MODELS = [
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
]
SUPPORTED_EMBEDDING_MODELS = [
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large",
]
SUPPORTED_RESPONSES_MODELS = (
    [f"responses:{m}" for m in SUPPORTED_CHAT_MODELS + SUPPORTED_REASONING_MODELS]
    + [f"responses:{m}" for m in SUPPORTED_RESPONSES_REASONING_MODELS]
    + [
        "responses:gpt-5-pro",
        "responses:o3-pro",
        "responses:gpt-5.2-pro",
        "responses:gpt-5.4-pro",
    ]
)


class OpenAIMixin:
    def apply_cache_breakpoints_to_messages(self, messages: list[dict], model: str) -> list[dict]:
        supported = model in SUPPORTED_RESPONSES_REASONING_MODELS
        prepared_messages = []
        breakpoint_count = 0
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                if CACHE_BREAKPOINT in content:
                    if not supported:
                        # This model cannot honor explicit breakpoints; strip the reserved
                        # control token so it never leaks into the request.
                        content = strip_cache_breakpoints(content)
                    else:
                        breakpoint_count += content.count(CACHE_BREAKPOINT)
                        if breakpoint_count > 4:
                            msg = "OpenAI supports at most four cache breakpoint writes per request."
                            raise ValueError(msg)
                        content = build_cache_breakpoint_blocks(content)
                prepared_messages.append({**message, "content": content})
                continue

            blocks = []
            for block in content:
                if block["type"] == "input_text" and CACHE_BREAKPOINT in block["text"]:
                    if not supported:
                        blocks.append({**block, "text": strip_cache_breakpoints(block["text"])})
                    else:
                        breakpoint_count += block["text"].count(CACHE_BREAKPOINT)
                        if breakpoint_count > 4:
                            msg = "OpenAI supports at most four cache breakpoint writes per request."
                            raise ValueError(msg)
                        blocks.extend(build_cache_breakpoint_blocks(block["text"]))
                else:
                    blocks.append(block)
            prepared_messages.append({**message, "content": blocks})
        return prepared_messages

    def api_max_context_tokens(self):
        if (
            self.model == "text-curie-001"
            or self.model == "text-babbage-001"
            or self.model == "text-ada-001"
            or self.model == "davinci"
            or self.model == "curie"
            or self.model == "babbage"
            or self.model == "ada"
        ):
            return 2_049
        if (
            self.model == "gpt-3.5-turbo"
            or self.model == "gpt-3.5-turbo-0613"
            or self.model == "gpt-3.5-turbo-1106"
        ):
            return 4_096
        if (
            self.model == "gpt-4"
            or self.model == "gpt-4-0613"
            or self.model == "text-embedding-ada-002"
            or self.model == "text-embedding-3-small"
            or self.model == "text-embedding-3-large"
        ):
            return 8_192
        if (
            self.model == "gpt-3.5-turbo-16k"
            or self.model == "gpt-3.5-turbo-16k-0613"
            or self.model == "davinci-002"
        ):
            return 16_384
        if self.model == "gpt-4-32k" or self.model == "gpt-4-32k-0613":
            return 32_768
        if (
            self.model == "gpt-4-1106-preview"
            or self.model == "gpt-4-turbo-2024-04-09"
            or self.model == "gpt-4-turbo"
            or self.model == "gpt-4-1106"
            or self.model == "gpt-4o"
            or self.model == "gpt-4o-2024-11-20"
            or self.model == "gpt-4o-mini"
            or self.model == "chatgpt-4o-latest"
            or self.model == "gpt-5.2-chat-latest"
        ):
            return 128_000
        if (
            self.model == "o1"
            or self.model == "o3"
            or self.model == "o3-mini"
            or self.model == "o3-pro"
            or self.model == "o4-mini"
            or self.model == "gpt-5-chat-latest"
            or self.model == "gpt-5.1-chat-latest"
        ):
            return 200_000
        if (
            self.model == "gpt-5"
            or self.model == "gpt-5.1"
            or self.model == "gpt-5.2"
            or self.model == "gpt-5-mini"
            or self.model == "gpt-5-nano"
            or self.model == "gpt-5-pro"
            or self.model == "gpt-5.2-pro"
            or self.model == "gpt-5.4-mini"
            or self.model == "gpt-5.4-nano"
        ):
            return 400_000
        if self.model == "gpt-4.1" or self.model == "gpt-4.1-mini" or self.model == "gpt-4.1-nano":
            return 1_047_576
        if (
            self.model == "gpt-5.4"
            or self.model == "gpt-5.4-pro"
            or self.model == "gpt-5.5"
            or self.model == "gpt-5.5-2026-04-23"
            or self.model == "gpt-5.6-sol"
            or self.model == "gpt-5.6-terra"
            or self.model == "gpt-5.6-luna"
        ):
            return 1_050_000
        msg = f"Unsupported model: {self.model}"
        UserMessage(msg)
        raise ValueError(msg)

    def api_max_response_tokens(self):
        if self.model == "davinci-002":
            return 2_048
        if (
            self.model == "gpt-4-turbo"
            or self.model == "gpt-4-turbo-2024-04-09"
            or self.model == "gpt-4-1106-preview"
            or self.model == "gpt-3.5-turbo-1106"
            or self.model == "gpt-3.5-turbo-0613"
            or self.model == "gpt-3.5-turbo"
        ):
            return 4_096
        if self.model == "gpt-4-0613" or self.model == "gpt-4":
            return 8_192
        if (
            self.model == "gpt-3.5-turbo-16k-0613"
            or self.model == "gpt-3.5-turbo-16k"
            or self.model == "gpt-4o-mini"
            or self.model == "gpt-4o"
            or self.model == "gpt-4o-2024-11-20"
            or self.model == "chatgpt-4o-latest"
            or self.model == "gpt-5-chat-latest"
            or self.model == "gpt-5.1-chat-latest"
            or self.model == "gpt-5.2-chat-latest"
        ):
            return 16_384
        if self.model == "gpt-4.1" or self.model == "gpt-4.1-mini" or self.model == "gpt-4.1-nano":
            return 32_768
        if (
            self.model == "o1"
            or self.model == "o3"
            or self.model == "o3-mini"
            or self.model == "o3-pro"
            or self.model == "o4-mini"
        ):
            return 100_000
        if (
            self.model == "gpt-5"
            or self.model == "gpt-5.1"
            or self.model == "gpt-5.2"
            or self.model == "gpt-5-mini"
            or self.model == "gpt-5-nano"
            or self.model == "gpt-5.2-pro"
            or self.model == "gpt-5.4"
            or self.model == "gpt-5.4-pro"
            or self.model == "gpt-5.4-mini"
            or self.model == "gpt-5.4-nano"
            or self.model == "gpt-5.5"
            or self.model == "gpt-5.5-2026-04-23"
            or self.model == "gpt-5.6-sol"
            or self.model == "gpt-5.6-terra"
            or self.model == "gpt-5.6-luna"
        ):
            return 128_000
        if self.model == "gpt-5-pro":
            return 272_000
        msg = f"Unsupported model: {self.model}"
        UserMessage(msg)
        raise ValueError(msg)

    def api_embedding_dims(self):
        if self.model == "text-embedding-ada-002":
            return 1_536
        if self.model == "text-embedding-3-small":
            return 1_536
        if self.model == "text-embedding-3-large":
            return 3_072
        msg = f"Unsupported model: {self.model}"
        UserMessage(msg)
        raise ValueError(msg)
