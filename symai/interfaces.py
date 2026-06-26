import logging
from functools import cache
from importlib.metadata import entry_points
from pydoc import locate

from .backend.mixin import OPENAI_CHAT_MODELS, OPENAI_REASONING_MODELS
from .backend.settings import SYMAI_CONFIG
from .imports import Import
from .symbol import Expression

logger = logging.getLogger(__name__)


@cache
def _expression_eps():
    """Installed `symai.expressions` plugins, keyed by entry-point name.

    Cached because `entry_points()` scans all distribution metadata and
    `Interface(...)` is a hot path. Names are a flat namespace across every
    installed distribution, so duplicates are resolved first-wins, loudly."""
    eps = {}
    for ep in entry_points(group="symai.expressions"):
        if ep.name in eps:
            logger.warning("Duplicate symai.expressions plugin %r; keeping first.", ep.name)
            continue

        eps[ep.name] = ep
    return eps


def load_expression(name: str):
    """Resolve an installed `symai.expressions` plugin to its class, or None.

    None means "not installed under that name" — callers fall back to the
    bundled `symai.extended.interfaces` registry."""
    ep = _expression_eps().get(name)
    if ep is None:
        return None

    return ep.load()


class Interface(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def __new__(cls, module: str, *args, **kwargs):
        module = str(module)
        # if `/` in module, assume github repo; else assume local module
        if "/" in module:
            return Import(module)
        name = module.lower().replace("-", "_")
        cls._module = name
        cls.module_path = f"symai.extended.interfaces.{name}"
        expression_cls = Interface.load_module_class(cls.module_path, name) or load_expression(name)
        if expression_cls is None:
            msg = f"No interface or installed symai.expressions plugin named {name!r}."
            raise ValueError(msg)

        return expression_cls(*args, **kwargs)

    def __call__(self, *_args, **_kwargs):
        msg = f"Interface {self._module} is not callable."
        raise NotImplementedError(msg)

    @staticmethod
    def load_module_class(module_path, class_name):
        module_ = locate(module_path)
        if module_ is None:
            return None

        return getattr(module_, class_name, None)


def _add_symbolic_interface(mapping):
    symbolic_api_key = SYMAI_CONFIG.get("SYMBOLIC_ENGINE_API_KEY")
    if symbolic_api_key is not None:
        mapping["symbolic"] = Interface("wolframalpha")


def _resolve_drawing_interface_name(drawing_engine_model):
    if drawing_engine_model.startswith("flux"):
        return "flux"
    if drawing_engine_model.startswith(("gemini-2.5-flash-image", "gemini-3-pro-image-preview")):
        return "nanobanana"
    if drawing_engine_model.startswith("dall-e-"):
        return "dall_e"
    if drawing_engine_model.startswith("gpt-image-"):
        return "gpt_image"
    return None


def _add_drawing_interface(mapping):
    drawing_engine_api_key = SYMAI_CONFIG.get("DRAWING_ENGINE_API_KEY")
    if drawing_engine_api_key is None:
        return
    drawing_engine_model = SYMAI_CONFIG.get("DRAWING_ENGINE_MODEL")
    interface_name = _resolve_drawing_interface_name(drawing_engine_model)
    if interface_name is not None:
        mapping["drawing"] = Interface(interface_name)


def _resolve_search_interface_name(search_engine_model):
    if search_engine_model.startswith("google"):
        return "serpapi"
    if search_engine_model.startswith("sonar"):
        return "perplexity"
    if search_engine_model in OPENAI_REASONING_MODELS + OPENAI_CHAT_MODELS:
        return "openai_search"
    return None


def _add_search_interface(mapping):
    search_engine_api_key = SYMAI_CONFIG.get("SEARCH_ENGINE_API_KEY")
    if search_engine_api_key is None:
        return
    search_engine_model = SYMAI_CONFIG.get("SEARCH_ENGINE_MODEL")
    interface_name = _resolve_search_interface_name(search_engine_model)
    if interface_name is not None:
        mapping["search"] = Interface(interface_name)


def _add_formal_interface(mapping):
    formal_api_key = SYMAI_CONFIG.get("FORMAL_ENGINE_API_KEY")
    if not formal_api_key:
        return
    engine_name = SYMAI_CONFIG.get("FORMAL_ENGINE", "axiom")
    mapping["formal"] = Interface(engine_name)


def _add_ocr_interface(mapping):
    ocr_api_key = SYMAI_CONFIG.get("OCR_ENGINE_API_KEY")
    if not ocr_api_key:
        return
    mapping["ocr"] = Interface("ocr")


def _add_tts_interface(mapping):
    tts_engine_api_key = SYMAI_CONFIG.get("TEXT_TO_SPEECH_ENGINE_API_KEY")
    if tts_engine_api_key is not None:  # TODO: add tests for this engine
        mapping["tts"] = Interface("tts")


def cfg_to_interface():
    """Maps configuration to interface."""
    mapping = {}
    _add_symbolic_interface(mapping)
    _add_formal_interface(mapping)
    _add_drawing_interface(mapping)
    _add_search_interface(mapping)
    _add_ocr_interface(mapping)
    _add_tts_interface(mapping)

    mapping["indexing"] = Interface("naive_vectordb")
    mapping["scraper"] = Interface("naive_scrape")
    mapping["stt"] = Interface("whisper")
    mapping["file"] = Interface("file")

    return mapping
