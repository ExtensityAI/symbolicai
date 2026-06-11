import numpy as np

from ...base import Engine
from ...settings import SYMAI_CONFIG
from ....interfaces import Interface

class LocalEmbeddingEngine(Engine):
    def __init__(self, model: str | None = None):
        super().__init__()
        self.config = SYMAI_CONFIG
        self.model_name = model or self.config.get("EMBEDDING_ENGINE_MODEL", "local")

        actual_model_name = self.model_name.replace("local/", "")

        if actual_model_name == "local" or actual_model_name == "":
            self._local_model = Interface("ExtensityAI/embeddings")
        else:
            # Pass the custom model name directly to the ExtensityAI constructor
            self._local_model = Interface("ExtensityAI/embeddings", model=actual_model_name)
        self.embedding_dim = 384 # It gets dynamically updated

    def id(self) -> str:
        if (
            not self.model_name
            or self.model_name == "local"
            or self.model_name.startswith("local/")
        ):
            return "embedding"
        return super().id()

    def forward(self, argument):
        prepared_input = argument.prop.prepared_input
        kwargs = argument.kwargs

        inp = prepared_input if isinstance(prepared_input, list) else [prepared_input]
        new_dim = kwargs.get("new_dim")

        output = self._local_model(inp)

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
            output = [self._normalize_l2(emb[:mn]) for emb in output if emb is not None]

        metadata = {"raw_output": output}
        return [output], metadata


    def prepare(self, argument):
        assert not argument.prop.processed_input, (
            "LocalEmbeddingEngine does not support processed_input."
        )
        argument.prop.prepared_input = argument.prop.entries

    def _normalize_l2(self, x):
        x = np.array(x)
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            if norm == 0:
                return x.tolist()
            return (x / norm).tolist()
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm).tolist()
