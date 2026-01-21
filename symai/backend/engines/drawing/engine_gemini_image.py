import base64
import logging
import mimetypes
import tempfile
from pathlib import Path

from google import genai
from google.genai import types

from ....symbol import Result
from ....utils import UserMessage
from ...base import Engine
from ...settings import SYMAI_CONFIG

logging.getLogger("google.genai").setLevel(logging.ERROR)
logging.getLogger("google_genai").propagate = False


class GeminiImageResult(Result):
    def __init__(self, value, **kwargs):
        super().__init__(value, **kwargs)
        paths = []
        for candidate in getattr(value, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", []) if content else []
            for part in parts:
                inline_data = getattr(part, "inline_data", None)
                if inline_data is None:
                    continue
                mime_type = getattr(inline_data, "mime_type", None) or "image/png"
                data = getattr(inline_data, "data", None)
                if data is None:
                    continue
                if isinstance(data, str):
                    data = base64.b64decode(data)
                suffix = mimetypes.guess_extension(mime_type) or ".png"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                    path = tmp_file.name
                with Path(path).open("wb") as f:
                    f.write(data)
                paths.append(path)
        if not paths:
            UserMessage("Gemini image generation returned no images.", raise_with=ValueError)
        self._value = paths


class GeminiImageEngine(Engine):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.config = SYMAI_CONFIG
        self.api_key = self.config.get("DRAWING_ENGINE_API_KEY") if api_key is None else api_key
        self.model = self.config.get("DRAWING_ENGINE_MODEL") if model is None else model
        self.name = self.__class__.__name__
        self.client = genai.Client(api_key=self.api_key)

    def id(self) -> str:
        cfg_model = self.config.get("DRAWING_ENGINE_MODEL")
        if cfg_model and cfg_model.startswith(("gemini-2.5-flash-image", "gemini-3-pro-image-preview")):
            return "drawing"
        return super().id()

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "DRAWING_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["DRAWING_ENGINE_API_KEY"]
            self.client = genai.Client(api_key=self.api_key)
        if "DRAWING_ENGINE_MODEL" in kwargs:
            self.model = kwargs["DRAWING_ENGINE_MODEL"]

    def prepare(self, argument):
        argument.prop.prepared_input = str(argument.prop.processed_input)

    def forward(self, argument):
        prompt = argument.prop.prepared_input
        kwargs = argument.kwargs
        model = kwargs.get("model", self.model)
        operation = kwargs.get("operation")

        if operation != "create":
            UserMessage(f"Unknown operation: {operation}", raise_with=ValueError)

        response_modalities = kwargs.get("response_modalities", ["IMAGE"])
        config = kwargs.get("config")
        if config is None:
            config = types.GenerateContentConfig(response_modalities=response_modalities)

        except_remedy = kwargs.get("except_remedy", None)
        try:
            res = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            if except_remedy is None:
                raise
            res = except_remedy(self, e, None, argument)

        metadata = {}
        result = GeminiImageResult(res)
        return [result], metadata
