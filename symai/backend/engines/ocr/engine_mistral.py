import logging
from copy import deepcopy

try:
    from mistralai.client import Mistral
except ImportError:
    Mistral = None

from ....symbol import Result
from ....utils import UserMessage
from ...base import Engine
from ...settings import SYMAI_CONFIG

logger = logging.getLogger(__name__)
# silence httpx debug noise from mistralai SDK
logging.getLogger("httpx").setLevel(logging.ERROR)


class MistralOCRResult(Result):
    """Result wrapper for Mistral OCR API responses."""

    def __init__(self, value, per_page: bool = False, **kwargs):
        raw = value.model_dump()
        super().__init__(raw, **kwargs)
        pages = raw["pages"]
        if per_page:
            self._value = [page["markdown"] for page in pages]
        else:
            self._value = "\n\n".join(page["markdown"] for page in pages)
        # build image mapping: id -> base64 data URI (only populated when include_image_base64=True)
        self._images = {}
        for page in pages:
            for img in page["images"]:
                b64 = img.get("image_base64")
                if b64:
                    self._images[img["id"]] = b64

    @property
    def images(self) -> dict[str, str]:
        """Mapping of image id to base64 data URI. Empty when include_image_base64 was not set."""
        return self._images

    def __str__(self) -> str:
        if isinstance(self._value, list):
            return "\n\n---\n\n".join(self._value)
        return self._value or ""


class MistralOCREngine(Engine):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        self.api_key = api_key or self.config.get("OCR_ENGINE_API_KEY")
        self.model = model or self.config.get("OCR_ENGINE_MODEL", "mistral-ocr-latest")
        self.name = self.__class__.__name__

        if self.id() == super().id():
            return

        if Mistral is None:
            UserMessage(
                "mistralai SDK is not installed. "
                "Install with 'pip install symbolicai[ocr]' or 'pip install mistralai'.",
                raise_with=ImportError,
            )

        self.client = Mistral(api_key=self.api_key)

    def id(self) -> str:
        if (
            self.config.get("OCR_ENGINE_API_KEY")
            and self.config.get("OCR_ENGINE_MODEL", "").lower().startswith("mistral")
        ):
            return "ocr"
        return super().id()

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "OCR_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["OCR_ENGINE_API_KEY"]
            self.client = Mistral(api_key=self.api_key)
        if "OCR_ENGINE_MODEL" in kwargs:
            self.model = kwargs["OCR_ENGINE_MODEL"]

    def prepare(self, argument):
        assert not argument.prop.processed_input, "MistralOCREngine does not support processed_input."
        document_url = getattr(argument.prop, "document_url", None)
        image_url = getattr(argument.prop, "image_url", None)
        assert document_url or image_url, "MistralOCREngine requires 'document_url' or 'image_url'."
        argument.prop.prepared_input = str(document_url or image_url)

    def forward(self, argument):
        kwargs = argument.kwargs
        per_page = kwargs.get("per_page", False)

        document_url = getattr(argument.prop, "document_url", None)
        image_url = getattr(argument.prop, "image_url", None)

        assert document_url or image_url, "Provide document_url or image_url."

        if document_url:
            document = {"type": "document_url", "document_url": str(document_url)}
        else:
            document = {"type": "image_url", "image_url": str(image_url)}

        ocr_kwargs = {"model": self.model, "document": document}

        # pass through Mistral-specific options from kwargs
        for key in ("table_format", "extract_header", "extract_footer", "include_image_base64"):
            if key in kwargs:
                ocr_kwargs[key] = kwargs[key]

        try:
            result = self.client.ocr.process(**ocr_kwargs)
        except Exception as e:
            UserMessage(f"Mistral OCR request failed: {e}", raise_with=RuntimeError)

        rsp = MistralOCRResult(result, per_page=per_page)
        metadata = {"raw_output": result}

        return [rsp], metadata
