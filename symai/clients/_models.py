from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

ModelId = Annotated[str, Field(min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class TolerantModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
