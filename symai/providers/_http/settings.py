from typing import Annotated

from pydantic import Field, SecretStr

from symai.providers._http.schema import StrictModel

PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class HttpProviderSettings(StrictModel):
    api_key: SecretStr = Field(min_length=1)
    model: str = Field(min_length=1)
    request_timeout: PositiveFiniteFloat = 600.0
    connect_timeout: PositiveFiniteFloat = 10.0
    connect_retries: int = Field(default=0, ge=0)
