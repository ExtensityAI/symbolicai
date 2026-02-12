import base64
import logging
import tempfile
import types as _types
from enum import Enum
from pathlib import Path

from box import Box, BoxList

from ....utils import UserMessage
from ...base import Engine

try:
    from markitdown import PRIORITY_GENERIC_FILE_FORMAT, PRIORITY_SPECIFIC_FILE_FORMAT, MarkItDown
    from markitdown.converters._audio_converter import AudioConverter
    from markitdown.converters._csv_converter import CsvConverter
    from markitdown.converters._docx_converter import DocxConverter
    from markitdown.converters._epub_converter import EpubConverter
    from markitdown.converters._html_converter import HtmlConverter
    from markitdown.converters._image_converter import ImageConverter
    from markitdown.converters._ipynb_converter import IpynbConverter
    from markitdown.converters._outlook_msg_converter import OutlookMsgConverter
    from markitdown.converters._pdf_converter import PdfConverter
    from markitdown.converters._plain_text_converter import PlainTextConverter
    from markitdown.converters._pptx_converter import PptxConverter
    from markitdown.converters._rss_converter import RssConverter
    from markitdown.converters._xlsx_converter import XlsConverter, XlsxConverter
    from markitdown.converters._zip_converter import ZipConverter

    _MARKITDOWN_AVAILABLE = True
except ImportError:
    _MARKITDOWN_AVAILABLE = False

logger = logging.getLogger(__name__)
for _noisy in ("pdfminer", "charset_normalizer", "PIL", "pydub"):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)

_SPECIFIC_CONVERTERS = (
    PdfConverter, DocxConverter, PptxConverter,
    XlsxConverter, XlsConverter,
    ImageConverter, AudioConverter,
    IpynbConverter, OutlookMsgConverter,
    EpubConverter, CsvConverter, RssConverter,
) if _MARKITDOWN_AVAILABLE else ()

_GENERIC_CONVERTERS = (
    PlainTextConverter, HtmlConverter,
) if _MARKITDOWN_AVAILABLE else ()


class _SymaiVisionClient:
    """Adapter: SymAI neurosymbolic vision -> markitdown's OpenAI-style llm_client.

    markitdown converters call client.chat.completions.create(model, messages)
    with OpenAI-format vision messages (base64 data URIs). This adapter extracts
    the data URI + prompt, routes through Symbol('<<vision:data_uri:>>').query(prompt)
    which works across all SymAI backends (OpenAI, Anthropic, Google, etc.),
    and wraps the result in an OpenAI-compatible response object.
    """

    class _Completions:
        @staticmethod
        def create(*, model, messages):  # noqa: ARG004
            from ....symbol import Symbol  # noqa: PLC0415

            # Extract text prompt and image data URI from OpenAI-format vision messages
            prompt, data_uri = "", ""
            for msg in messages:
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if part.get("type") == "text":
                        prompt = part["text"]
                    elif part.get("type") == "image_url":
                        data_uri = part["image_url"]["url"]
            # Not all engines handle inline data URIs (e.g. Anthropic calls
            # encode_media_frames which expects a file path). Decode to a temp
            # file so every backend can process it uniformly.
            img_ref = data_uri
            if data_uri.startswith("data:image"):
                # data:image/<fmt>;base64,<payload> → extract format + raw bytes
                header, payload = data_uri.split(",", 1)
                # Matches e.g. "data:image/png;base64" → ext = "png"
                ext = header.split("/")[1].split(";")[0]
                with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
                    tmp.write(base64.b64decode(payload))
                img_ref = tmp.name
            try:
                result = Symbol(f"<<vision:{img_ref}:>>").query(prompt)
                return _types.SimpleNamespace(
                    choices=[_types.SimpleNamespace(
                        message=_types.SimpleNamespace(content=result.value)
                    )]
                )
            finally:
                if img_ref != data_uri:
                    Path(img_ref).unlink(missing_ok=True)

    def __init__(self):
        self.chat = _types.SimpleNamespace(completions=self._Completions())


class SupportedFileType(Enum):
    """File types supported by the reader engine."""

    # Plain text -- read directly via native Python I/O
    TXT = ".txt"
    MD = ".md"
    PY = ".py"
    JSON = ".json"
    YAML = ".yaml"
    YML = ".yml"
    CSV = ".csv"
    TSV = ".tsv"
    TOML = ".toml"
    LOG = ".log"
    # Rich formats -- converted via markitdown
    PDF = ".pdf"
    DOCX = ".docx"
    PPTX = ".pptx"
    XLSX = ".xlsx"
    XLS = ".xls"
    HTML = ".html"
    HTM = ".htm"
    XML = ".xml"
    EPUB = ".epub"
    IPYNB = ".ipynb"
    ZIP = ".zip"
    # Images -- markitdown extracts EXIF metadata + optional LLM caption
    JPG = ".jpg"
    JPEG = ".jpeg"
    PNG = ".png"
    # Audio -- markitdown extracts metadata + optional speech transcription
    MP3 = ".mp3"
    WAV = ".wav"
    M4A = ".m4a"
    MP4 = ".mp4"


_PLAIN_TEXT_EXTS = {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv", ".tsv", ".toml", ".xml", ".log"}
_RICH_FORMAT_EXTS = {ft.value for ft in SupportedFileType} - _PLAIN_TEXT_EXTS

_BOX_PARSERS = {
    ".json": lambda s: Box.from_json(json_string=s),
    ".yaml": lambda s: Box.from_yaml(yaml_string=s),
    ".yml": lambda s: Box.from_yaml(yaml_string=s),
    ".toml": lambda s: Box.from_toml(toml_string=s),
    ".csv": lambda s: BoxList.from_csv(csv_string=s),
}


class FileEngine(Engine):
    def __init__(self):
        super().__init__()
        self._converter = None  # lazy MarkItDown instance

    def id(self) -> str:
        return "files"

    def _get_converter(self):
        """Lazily build a MarkItDown instance with selective converters + LLM config."""
        if self._converter is not None:
            return self._converter
        if not _MARKITDOWN_AVAILABLE:
            UserMessage(
                "markitdown is not installed. Install with: pip install 'symbolicai[files]'",
                raise_with=ImportError,
            )
        md = MarkItDown(enable_builtins=False)

        # Wire SymAI vision adapter for LLM-powered converters (image captions, PPTX)
        from ...settings import SYMAI_CONFIG  # noqa: PLC0415

        md._llm_client = _SymaiVisionClient()
        md._llm_model = SYMAI_CONFIG.get("NEUROSYMBOLIC_ENGINE_MODEL", "")

        for conv_cls in _SPECIFIC_CONVERTERS:
            md.register_converter(conv_cls(), priority=PRIORITY_SPECIFIC_FILE_FORMAT)
        for conv_cls in _GENERIC_CONVERTERS:
            md.register_converter(conv_cls(), priority=PRIORITY_GENERIC_FILE_FORMAT)
        # ZipConverter needs a back-reference for recursive conversion of archive contents
        md.register_converter(ZipConverter(markitdown=md), priority=PRIORITY_GENERIC_FILE_FORMAT)

        self._converter = md
        return md

    def _read_plain_text(self, path_obj):
        """Read file as plain text via native Python I/O."""
        with path_obj.open(encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _read_via_markitdown(self, source):
        """Convert file (or URL) to Markdown text via markitdown converters."""
        return self._get_converter().convert(str(source)).text_content

    def forward(self, argument):
        path = argument.prop.prepared_input
        if path is None or path.strip() == "":
            return [None], {}

        backend = argument.kwargs.get("backend", "standard")

        # URLs require markitdown (HTML fetch + conversion)
        if path.startswith("http://") or path.startswith("https://"):
            if backend != "markitdown":
                UserMessage(
                    f"URLs require backend='markitdown'. Got backend='{backend}'.",
                    raise_with=ValueError,
                )
            rsp = self._read_via_markitdown(path)
            if rsp is None:
                UserMessage(f"Error reading URL - empty result: {path}", raise_with=Exception)
            return [rsp], {}
        path_obj = Path(path)

        assert path_obj.exists(), f"File does not exist: {path}"
        if path_obj.stat().st_size <= 0:
            return [""], {}

        ext = path_obj.suffix.lower()
        all_supported = {ft.value for ft in SupportedFileType}

        if backend == "markitdown":
            if ext not in all_supported:
                supported = ", ".join(sorted(all_supported))
                UserMessage(
                    f"Extension '{ext}' is not supported by the markitdown backend. "
                    f"Supported: {supported}",
                    raise_with=ValueError,
                )
            rsp = self._read_via_markitdown(path_obj)
        else:
            if ext not in _PLAIN_TEXT_EXTS:
                if ext in _RICH_FORMAT_EXTS:
                    UserMessage(
                        f"Extension '{ext}' is not supported by the standard backend. "
                        f"Try changing the backend from 'standard' to 'markitdown'.",
                        raise_with=ValueError,
                    )
                supported = ", ".join(sorted(all_supported))
                UserMessage(
                    f"Unsupported file extension '{ext}'. Supported: {supported}",
                    raise_with=ValueError,
                )
            rsp = self._read_plain_text(path_obj)

        if rsp is None:
            UserMessage(f"Error reading file - empty result: {path}", raise_with=Exception)

        if argument.kwargs.get("as_box", False):
            parser = _BOX_PARSERS.get(ext)
            if parser is None:
                supported = ", ".join(sorted(_BOX_PARSERS))
                UserMessage(
                    f"as_box is not supported for '{ext}'. "
                    f"Supported: {supported}",
                    raise_with=ValueError,
                )
            rsp = parser(rsp)

        return [rsp], {}

    def prepare(self, argument):
        assert not argument.prop.processed_input, "FileEngine does not support processed_input."
        raw = argument.prop.path
        if raw.startswith(("http://", "https://")):
            argument.prop.prepared_input = raw
        else:
            argument.prop.prepared_input = Path(raw).as_posix()
