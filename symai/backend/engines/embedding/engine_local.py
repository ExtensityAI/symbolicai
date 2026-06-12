from copy import deepcopy

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from ....symbol import Symbol
from ....utils import UserMessage
from ...base import Engine
from ...settings import SYMAI_CONFIG


class LocalEmbeddingEngine(Engine):
    DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"

    def __init__(self, model: str | None = None):
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        self.model_name = model or self.config.get("EMBEDDING_ENGINE_MODEL") or self.DEFAULT_MODEL
        if SentenceTransformer is None:
            UserMessage(
                "sentence-transformers is not installed. Install with: pip install symbolicai[hf]",
                raise_with=ImportError,
            )
        self.model = SentenceTransformer(self.model_name.replace("local:", ""))
        self.embedding_dim = 384  # NOTE: Gets dynamically updated in forward()
        self.name = self.__class__.__name__

    def id(self) -> str:
        if (
            not self.model_name
            or self.model_name.startswith("local:")
            or self.model_name == self.DEFAULT_MODEL
        ):
            return "embedding"
        return super().id()

    def forward(self, argument):
        prepared_input = argument.prop.prepared_input
        kwargs = argument.kwargs

        inp = prepared_input.value if isinstance(prepared_input, Symbol) else prepared_input
        if not isinstance(inp, list):
            inp = [inp]
        new_dim = kwargs.get("new_dim")

        output = self.model.encode(
            inp,
            convert_to_numpy=True,
        )

        if hasattr(output, "value"):
            output = output.value
        if hasattr(output, "tolist"):
            output = output.tolist()
        elif hasattr(output, "shape") and len(output.shape) == 1:
            output = [output.tolist()]

        # Update embedding_dim based on returned vector length
        if output and len(output) > 0:
            self.embedding_dim = len(output[0])

        if new_dim:
            mn = min(new_dim, self.embedding_dim)
            output = [self.normalize_l2(emb[:mn]) for emb in output if emb is not None]

        metadata = {"raw_output": output}
        return [output], metadata

    def prepare(self, argument):
        assert not argument.prop.processed_input, (
            "LocalEmbeddingEngine does not support processed_input."
        )
        argument.prop.prepared_input = argument.prop.entries

    def normalize_l2(self, x):
        x = np.array(x)
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            if norm == 0:
                return x.tolist()
            return (x / norm).tolist()
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm).tolist()
