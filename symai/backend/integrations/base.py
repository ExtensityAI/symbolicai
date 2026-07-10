from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class TolerantModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
