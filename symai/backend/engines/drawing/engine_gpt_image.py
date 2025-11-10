import base64
import contextlib
import logging
import tempfile
from pathlib import Path

import openai
import requests

from ....symbol import Result
from ....utils import UserMessage
from ...base import Engine
from ...settings import SYMAI_CONFIG

# silence noisy libraries
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


class GPTImageResult(Result):
    """
    Wraps an OpenAI images.generate / edit / variation response.
    Exposes .value as the raw response and ._value as the
    first URL or decoded b64 image string.
    """
    def __init__(self, value, **kwargs):
        super().__init__(value, **kwargs)
        imgs = []
        for item in value.data:
            has_url = hasattr(item, "url")
            has_b64 = hasattr(item, "b64_json")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                path = tmp_file.name
            if has_url and item.url is not None:
                request = requests.get(item.url, allow_redirects=True)
                request.raise_for_status()
                with Path(path).open("wb") as f:
                    f.write(request.content)
            elif has_b64 and item.b64_json is not None:
                raw = base64.b64decode(item.b64_json)
                with Path(path).open("wb") as f:
                    f.write(raw)
            imgs.append(path)
        self._value = imgs


class GPTImageEngine(Engine):
    """
    A drop-in engine for OpenAI's unified Images API,
    supporting gpt-image-1, dall-e-2, dall-e-3,
    with all the extra parameters (background, moderation, etc).
    """
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        super().__init__()
        self.config = SYMAI_CONFIG
        # pick up a separate config slot if you like, or fall back
        openai.api_key = (
            self.config.get("DRAWING_ENGINE_API_KEY")
            if api_key is None
            else api_key
        )
        self.model = (
            self.config.get("DRAWING_ENGINE_MODEL")
            if model is None
            else model
        )
        self.name = self.__class__.__name__
        # quiet OpenAI's internal logger
        log = logging.getLogger("openai")
        log.setLevel(logging.WARNING)

    def id(self) -> str:
        # register this engine under "gpt-image" by default
        cfg_model = self.config.get("DRAWING_ENGINE_MODEL")
        if cfg_model.startswith("gpt-image-") or cfg_model.startswith("dall-e-"):
            return "drawing"
        return super().id()

    def command(self, *args, **kwargs):
        """
        Allow hot-swapping API key or model at runtime.
        """
        super().command(*args, **kwargs)
        if "DRAWING_ENGINE_API_KEY" in kwargs:
            openai.api_key = kwargs["DRAWING_ENGINE_API_KEY"]
        if "DRAWING_ENGINE_MODEL" in kwargs:
            self.model = kwargs["DRAWING_ENGINE_MODEL"]

    def prepare(self, argument):
        """
        Simply copy processed_input → prepared_input
        """
        argument.prop.prepared_input = str(argument.prop.processed_input)

    def forward(self, argument):
        prompt = argument.prop.prepared_input
        kwargs = argument.kwargs

        model = kwargs.get("model", self.model)
        operation = kwargs.get("operation")

        if operation is None:
            UserMessage("Operation not specified!", raise_with=ValueError)

        n = kwargs.get("n", 1)

        self._normalize_size(kwargs)

        except_remedy = kwargs.get("except_remedy", None)

        callback = None
        try:
            callback = self._resolve_callback(operation)
            callback, res = self._dispatch_operation(
                operation=operation,
                prompt=prompt,
                model=model,
                n=n,
                kwargs=kwargs,
            )
        except Exception as e:
            if except_remedy is None:
                raise
            res = except_remedy(self, e, callback, argument)

        metadata = {}
        result = GPTImageResult(res)
        return [result], metadata

    def _normalize_size(self, kwargs):
        if "size" in kwargs and isinstance(kwargs["size"], int):
            s = kwargs["size"]
            kwargs["size"] = f"{s}x{s}"

    def _resolve_callback(self, operation):
        if operation == "create":
            return openai.images.generate
        if operation == "variation":
            return openai.images.create_variation
        if operation == "edit":
            return openai.images.edit
        UserMessage(f"Unknown image operation: {operation}", raise_with=ValueError)
        return openai.images.generate

    def _dispatch_operation(self, operation, prompt, model, n, kwargs):
        if operation == "create":
            return self._execute_create(prompt, model, n, kwargs)
        if operation == "variation":
            return self._execute_variation(model, n, kwargs)
        if operation == "edit":
            return self._execute_edit(prompt, model, n, kwargs)
        return UserMessage(f"Unknown image operation: {operation}", raise_with=ValueError)

    def _execute_create(self, prompt, model, n, kwargs):
        create_kwargs = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": kwargs.get("size"),
        }

        if model == "dall-e-3":
            create_kwargs["response_format"] = kwargs.get("response_format", "url")
            create_kwargs["quality"] = kwargs.get("quality", "standard")
            create_kwargs["style"] = kwargs.get("style", "vivid")

        if model.startswith("gpt-image-"):
            create_kwargs["quality"] = kwargs.get("quality", "medium")
            create_kwargs["moderation"] = kwargs.get("moderation", "auto")
            create_kwargs["background"] = kwargs.get("background", "auto")
            create_kwargs["output_format"] = kwargs.get("output_compression", "png")
            if create_kwargs["output_format"] == "jpeg" or create_kwargs["output_format"] == "webp":
                create_kwargs["output_compression"] = kwargs.get("output_compression", "100")

        callback = openai.images.generate
        return callback, callback(**create_kwargs)

    def _execute_variation(self, model, n, kwargs):
        assert "image_path" in kwargs, "image_path required for variation"
        callback = openai.images.create_variation
        with Path(kwargs["image_path"]).open("rb") as img:
            result = callback(
                model=model,
                image=img,
                n=n,
                size=kwargs.get("size"),
                response_format=kwargs.get("response_format", "url"),
            )
        return callback, result

    def _execute_edit(self, prompt, model, n, kwargs):
        assert "image_path" in kwargs, "image_path required for edit"
        img_paths = kwargs["image_path"]
        if not isinstance(img_paths, (list, tuple)):
            img_paths = [img_paths]
        with contextlib.ExitStack() as stack:
            image_files = [stack.enter_context(Path(p).open("rb")) for p in img_paths]
            mask_file = None
            if "mask_path" in kwargs and kwargs["mask_path"] is not None:
                mask_file = stack.enter_context(Path(kwargs["mask_path"]).open("rb"))
            edit_kwargs = {
                "model": model,
                "image": image_files if len(image_files) > 1 else image_files[0],
                "prompt": prompt,
                "n": n,
                "size": kwargs.get("size"),
            }

            if model.startswith("gpt-image-"):
                edit_kwargs["quality"] = kwargs.get("quality", "auto")

            if mask_file:
                edit_kwargs["mask"] = mask_file
            callback = openai.images.edit
            result = callback(**edit_kwargs)
        return callback, result
