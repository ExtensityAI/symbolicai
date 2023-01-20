import os
import json
import logging


SYMAI_VERSION = "0.2.2"


def _start_symai():
    global _ai_config_
    # check if symai is already initialized
    _ai_config_path_ = os.path.join(os.getcwd(), 'symai.config.json')
    if not os.path.exists(_ai_config_path_):
        _parent_dir_ = os.path.dirname(__file__)
        _ai_config_path_ = os.path.join(_parent_dir_, 'symai.config.json')
    
    # read the symai.config.json file
    with open(_ai_config_path_, 'r') as f:
        _ai_config_ = json.load(f)
    _tmp_ai_config_ = _ai_config_.copy()

    # check which modules are already installed
    _openai_api_key_ = os.environ.get('OPENAI_API_KEY', None)
    
    _neurosymbolic_engine_api_key_ = os.environ.get('NEUROSYMBOLIC_ENGINE_API_KEY', None)
    _neurosymbolic_engine_model_ = os.environ.get('NEUROSYMBOLIC_ENGINE_MODEL', None)

    _symbolic_engine_api_key_ = os.environ.get('SYMBOLIC_ENGINE_API_KEY', None)

    _embedding_engine_api_key_ = os.environ.get('EMBEDDING_ENGINE_API_KEY', None)
    _embedding_engine_model_ = os.environ.get('EMBEDDING_ENGINE_MODEL', None)

    _imagerendering_engine_api_key_ = os.environ.get('IMAGERENDERING_ENGINE_API_KEY', None)

    _vision_engine_model_ = os.environ.get('VISION_ENGINE_MODEL', None)

    _search_engine_api_key_ = os.environ.get('SEARCH_ENGINE_API_KEY', None)
    _search_engine_model_ = os.environ.get('SEARCH_ENGINE_MODEL', None)

    _ocr_engine_api_key_ = os.environ.get('OCR_ENGINE_API_KEY', None)

    _speech_engine_model_ = os.environ.get('SPEECH_ENGINE_MODEL', None)
    
    _selenium_chrome_driver_version_ = os.environ.get('SELENIUM_CHROME_DRIVER_VERSION', None)

    # set or update the symai.config.json file
    if _neurosymbolic_engine_api_key_: _ai_config_['NEUROSYMBOLIC_ENGINE_API_KEY'] = _neurosymbolic_engine_api_key_
    if _symbolic_engine_api_key_: _ai_config_['SYMBOLIC_ENGINE_API_KEY'] = _symbolic_engine_api_key_
    if _embedding_engine_api_key_: _ai_config_['EMBEDDING_ENGINE_API_KEY'] = _embedding_engine_api_key_
    if _imagerendering_engine_api_key_: _ai_config_['IMAGERENDERING_ENGINE_API_KEY'] = _imagerendering_engine_api_key_
    
    # if user did not set the individual api keys, use the OPENAI_API_KEY environment variable
    if _openai_api_key_ and not _neurosymbolic_engine_api_key_: _ai_config_['NEUROSYMBOLIC_ENGINE_API_KEY'] = _openai_api_key_
    if _openai_api_key_ and not _embedding_engine_api_key_: _ai_config_['EMBEDDING_ENGINE_API_KEY'] = _openai_api_key_
    if _openai_api_key_ and not _imagerendering_engine_api_key_: _ai_config_['IMAGERENDERING_ENGINE_API_KEY'] = _openai_api_key_

    # set optional keys and modules
    if _neurosymbolic_engine_model_: _ai_config_['NEUROSYMBOLIC_ENGINE_MODEL'] = _neurosymbolic_engine_model_
    if _embedding_engine_model_: _ai_config_['EMBEDDING_ENGINE_MODEL'] = _embedding_engine_model_
    if _vision_engine_model_: _ai_config_['VISION_ENGINE_MODEL'] = _vision_engine_model_
    if _search_engine_api_key_: _ai_config_['SEARCH_ENGINE_API_KEY'] = _search_engine_api_key_
    if _search_engine_model_: _ai_config_['SEARCH_ENGINE_MODEL'] = _search_engine_model_
    if _ocr_engine_api_key_: _ai_config_['OCR_ENGINE_API_KEY'] = _ocr_engine_api_key_
    if _speech_engine_model_: _ai_config_['SPEECH_ENGINE_MODEL'] = _speech_engine_model_
    if _selenium_chrome_driver_version_: _ai_config_['SELENIUM_CHROME_DRIVER_VERSION'] = _selenium_chrome_driver_version_

    # verify if the symai.config.json file has changed
    _updated_key_ = {k: not _tmp_ai_config_[k] == _ai_config_[k] for k in _tmp_ai_config_.keys()}
    _has_changed_ = any(_updated_key_.values())

    if _has_changed_:
        # update the symai.config.json file
        with open(_ai_config_path_, 'w') as f:
            json.dump(_ai_config_, f, indent=4)
    
    # check if the mandatory keys are set
    with open(_ai_config_path_, 'r') as f:
        _ai_config_ = json.load(f)
    
    if _ai_config_['NEUROSYMBOLIC_ENGINE_API_KEY'] is None or len(_ai_config_['NEUROSYMBOLIC_ENGINE_API_KEY']) == 0:
        logging.error('The mandatory neuro-symbolic engine is not initialized. Please get a key from https://beta.openai.com/account/api-keys and set either a general environment variable OPENAI_API_KEY or a module specific environment variable NEUROSYMBOLIC_ENGINE_API_KEY.')

    import symai.backend.settings as settings
    settings.SYMAI_CONFIG = _ai_config_


_start_symai()


from .core import *
from .pre_processors import *
from .post_processors import *
from .symbol import *
from .components import *
from .prompts import Prompt
from .backend.base import Engine
from .functional import (_process_query, ConstraintViolationException)
from .chat import ChatBot
