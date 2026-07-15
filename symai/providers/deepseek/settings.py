from pydantic import Field, SecretStr

from symai.runtime.models import FrozenModel, PositiveFiniteFloat


class ChatCompletionsSettings(FrozenModel):
    api_key: SecretStr = Field(min_length=1)
    model: str = Field(min_length=1)
    request_timeout: PositiveFiniteFloat = 600.0
    connect_timeout: PositiveFiniteFloat = 10.0
    connect_retries: int = Field(default=0, ge=0)
