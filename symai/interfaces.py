import logging
from pydoc import locate

from .backend.mixin import OPENAI_CHAT_MODELS, OPENAI_REASONING_MODELS
from .backend.settings import SYMAI_CONFIG
from .imports import Import
from .symbol import Expression
from .utils import UserMessage


class Interface(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def __new__(cls, module: str, *args, **kwargs):
        module = str(module)
        # if `/` in module, assume github repo; else assume local module
        if "/" in module:
            return Import(module)
        module = module.lower()
        module = module.replace("-", "_")
        cls._module = module
        cls.module_path = f"symai.extended.interfaces.{module}"
        return Interface.load_module_class(cls.module_path, cls._module)(*args, **kwargs)

    def __call__(self, *_args, **_kwargs):
        UserMessage(f"Interface {self._module} is not callable.", raise_with=NotImplementedError)

    @staticmethod
    def load_module_class(module_path, class_name):
        module_ = locate(module_path)
        return getattr(module_, class_name)


def _add_symbolic_interface(mapping):
    symbolic_api_key = SYMAI_CONFIG.get("SYMBOLIC_ENGINE_API_KEY")
    if symbolic_api_key is not None:
        mapping["symbolic"] = Interface("wolframalpha")


def _resolve_drawing_interface_name(drawing_engine_model):
    if drawing_engine_model.startswith("flux"):
        return "flux"
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


def _add_tts_interface(mapping):
    tts_engine_api_key = SYMAI_CONFIG.get("TEXT_TO_SPEECH_ENGINE_API_KEY")
    if tts_engine_api_key is not None:  # TODO: add tests for this engine
        mapping["tts"] = Interface("tts")


def cfg_to_interface():
    """Maps configuration to interface."""
    mapping = {}
    _add_symbolic_interface(mapping)
    _add_drawing_interface(mapping)
    _add_search_interface(mapping)
    _add_tts_interface(mapping)

    mapping["indexing"] = Interface("naive_vectordb")
    mapping["scraper"] = Interface("naive_webscraping")
    mapping["stt"] = Interface("whisper")
    mapping["file"] = Interface("file")

    return mapping
