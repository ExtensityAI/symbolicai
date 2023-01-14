import os
import json
import logging


def _start_botdyn():
    global _bd_config_
    # check if botdyn is already initialized
    _bd_config_path_ = os.path.join(os.getcwd(), 'bd.config.json')
    if not os.path.exists(_bd_config_path_):
        _parent_dir_ = os.path.dirname(__file__)
        _bd_config_path_ = os.path.join(_parent_dir_, 'bd.config.json')
    
    # read the bd.config.json file
    with open(_bd_config_path_, 'r') as f:
        _bd_config_ = json.load(f)
    _tmp_bd_config_ = _bd_config_.copy()

    # check which modules are already installed
    _openai_api_key_ = os.environ.get('OPENAI_API_KEY', None)
    
    _neurosymbolic_engine_api_key_ = os.environ.get('NEUROSYMBOLIC_ENGINE_API_KEY', None)
    _neurosymbolic_engine_model_ = os.environ.get('NEUROSYMBOLIC_ENGINE_MODEL', None)

    _embedding_engine_api_key_ = os.environ.get('EMBEDDING_ENGINE_API_KEY', None)
    _embedding_engine_model_ = os.environ.get('EMBEDDING_ENGINE_MODEL', None)

    _imagerendering_engine_api_key_ = os.environ.get('IMAGERENDERING_ENGINE_API_KEY', None)

    _vision_engine_model_ = os.environ.get('VISION_ENGINE_MODEL', None)

    _search_engine_api_key_ = os.environ.get('SEARCH_ENGINE_API_KEY', None)
    _search_engine_model_ = os.environ.get('SEARCH_ENGINE_MODEL', None)

    _ocr_engine_api_key_ = os.environ.get('OCR_ENGINE_API_KEY', None)

    _speech_engine_model_ = os.environ.get('SPEECH_ENGINE_MODEL', None)
    
    _selenium_chrome_driver_version_ = os.environ.get('SELENIUM_CHROME_DRIVER_VERSION', None)

    # set or update the bd.config.json file
    if _neurosymbolic_engine_api_key_: _bd_config_['NEUROSYMBOLIC_ENGINE_API_KEY'] = _neurosymbolic_engine_api_key_
    if _embedding_engine_api_key_: _bd_config_['EMBEDDING_ENGINE_API_KEY'] = _embedding_engine_api_key_
    if _imagerendering_engine_api_key_: _bd_config_['IMAGERENDERING_ENGINE_API_KEY'] = _imagerendering_engine_api_key_
    
    # if user did not set the individual api keys, use the OPENAI_API_KEY environment variable
    if _openai_api_key_ and not _neurosymbolic_engine_api_key_: _bd_config_['NEUROSYMBOLIC_ENGINE_API_KEY'] = _openai_api_key_
    if _openai_api_key_ and not _embedding_engine_api_key_: _bd_config_['EMBEDDING_ENGINE_API_KEY'] = _openai_api_key_
    if _openai_api_key_ and not _imagerendering_engine_api_key_: _bd_config_['IMAGERENDERING_ENGINE_API_KEY'] = _openai_api_key_

    # set optional keys and modules
    if _neurosymbolic_engine_model_: _bd_config_['NEUROSYMBOLIC_ENGINE_MODEL'] = _neurosymbolic_engine_model_
    if _embedding_engine_model_: _bd_config_['EMBEDDING_ENGINE_MODEL'] = _embedding_engine_model_
    if _vision_engine_model_: _bd_config_['VISION_ENGINE_MODEL'] = _vision_engine_model_
    if _search_engine_api_key_: _bd_config_['SEARCH_ENGINE_API_KEY'] = _search_engine_api_key_
    if _search_engine_model_: _bd_config_['SEARCH_ENGINE_MODEL'] = _search_engine_model_
    if _ocr_engine_api_key_: _bd_config_['OCR_ENGINE_API_KEY'] = _ocr_engine_api_key_
    if _speech_engine_model_: _bd_config_['SPEECH_ENGINE_MODEL'] = _speech_engine_model_
    if _selenium_chrome_driver_version_: _bd_config_['SELENIUM_CHROME_DRIVER_VERSION'] = _selenium_chrome_driver_version_

    # verify if the bd.config.json file has changed
    _updated_key_ = {k: not _tmp_bd_config_[k] == _bd_config_[k] for k in _tmp_bd_config_.keys()}
    _has_changed_ = any(_updated_key_.values())

    if _has_changed_:
        # update the bd.config.json file
        with open(_bd_config_path_, 'w') as f:
            json.dump(_bd_config_, f, indent=4)
    
    # check if the mandatory keys are set
    with open(_bd_config_path_, 'r') as f:
        _bd_config_ = json.load(f)
    
    if _bd_config_['NEUROSYMBOLIC_ENGINE_API_KEY'] is None or len(_bd_config_['NEUROSYMBOLIC_ENGINE_API_KEY']) == 0:
        logging.error('The mandatory neuro-symbolic engine is not initialized. Please get a key from https://beta.openai.com/account/api-keys and set either a general environment variable OPENAI_API_KEY or a module specific environment variable NEUROSYMBOLIC_ENGINE_API_KEY.')

    import botdyn.backend.settings as settings
    settings.BOTDYN_CONFIG = _bd_config_


_start_botdyn()


from .core import *
from .pre_processors import *
from .post_processors import *
from .symbol import *
from .components import *
from .prompts import Prompt
from .backend.base import Engine
from .functional import (_process_query, ConstraintViolationException)
