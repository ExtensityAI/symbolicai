from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

ModelId = Annotated[str, Field(min_length=1)]


# `hide_input_in_errors` keeps a rejected value out of the ValidationError message, args,
# and traceback. Everything these models carry is something that must never reach a log:
# the API key, the prompt, and the provider's response body. The field name, location, and
# rule are still reported; only the value is withheld.
class StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", hide_input_in_errors=True)


class TolerantModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow", hide_input_in_errors=True)
