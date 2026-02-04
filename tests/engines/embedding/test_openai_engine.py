import pytest

from symai import Symbol
from symai.backend.settings import SYMAI_CONFIG

MODEL = SYMAI_CONFIG.get("EMBEDDING_ENGINE_MODEL", "")
IS_OPENAI = MODEL.startswith("text-embedding")
SKIP_MSG = f"EMBEDDING_ENGINE_MODEL={MODEL!r} is not an OpenAI model"


@pytest.mark.skipif(not IS_OPENAI, reason=SKIP_MSG)
def test_batched_embedding():
    result = Symbol(["hello", "world"]).embed()
    assert len(result.value) == 2


@pytest.mark.skipif(not IS_OPENAI, reason=SKIP_MSG)
def test_truncated_embedding():
    result = Symbol(["hello"]).embed(new_dim=128)
    assert len(result.value[0]) == 128
