import logging
from pydoc import locate

from .backend.mixin import OPENAI_CHAT_MODELS, OPENAI_REASONING_MODELS
from .backend.settings import SYMAI_CONFIG
from .imports import Import
from .symbol import Expression
from .utils import CustomUserWarning


class Interface(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def __new__(self, module: str, *args, **kwargs):
        module = str(module)
        # if `/` in module, assume github repo; else assume local module
        if "/" in module:
            return Import(module)
        module = module.lower()
        module = module.replace("-", "_")
        self._module = module
        self.module_path = f"symai.extended.interfaces.{module}"
        return Interface.load_module_class(self.module_path, self._module)(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        CustomUserWarning(f"Interface {self._module} is not callable.", raise_with=NotImplementedError)

    @staticmethod
    def load_module_class(module_path, class_name):
        module_ = locate(module_path)
        return getattr(module_, class_name)


def cfg_to_interface():
    """Maps configuration to interface."""
    mapping = {}
    symbolic_api_key = SYMAI_CONFIG.get("SYMBOLIC_ENGINE_API_KEY")
    if symbolic_api_key is not None:
        mapping["symbolic"] = Interface("wolframalpha")

    drawing_engine_api_key = SYMAI_CONFIG.get("DRAWING_ENGINE_API_KEY")
    if drawing_engine_api_key is not None:
        drawing_engine_model = SYMAI_CONFIG.get("DRAWING_ENGINE_MODEL")
        if drawing_engine_model.startswith("flux"):
            mapping["drawing"] = Interface("flux")
        elif drawing_engine_model.startswith("dall-e-"):
            mapping["drawing"] = Interface("dall_e")
        elif drawing_engine_model.startswith("gpt-image-"):
            mapping["drawing"] = Interface("gpt_image")

    search_engine_api_key = SYMAI_CONFIG.get("SEARCH_ENGINE_API_KEY")
    if search_engine_api_key is not None:
        search_engine_model = SYMAI_CONFIG.get("SEARCH_ENGINE_MODEL")
        if search_engine_model.startswith("google"):
            mapping["search"] = Interface("serpapi")
        elif search_engine_model.startswith("sonar"):
            mapping["search"] = Interface("perplexity")
        elif search_engine_model in OPENAI_REASONING_MODELS + OPENAI_CHAT_MODELS:
            mapping["search"] = Interface("openai_search")

    tts_engine_api_key = SYMAI_CONFIG.get("TEXT_TO_SPEECH_ENGINE_API_KEY")
    if tts_engine_api_key is not None: # TODO: add tests for this engine
        mapping["tts"] = Interface("tts")

    mapping["indexing"] = Interface("naive_vectordb")
    mapping["scraper"] = Interface("naive_webscraping")
    mapping["stt"] = Interface("whisper")
    mapping["file"] = Interface("file")

    return mapping
