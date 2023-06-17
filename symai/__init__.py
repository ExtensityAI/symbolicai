import json
import logging
import os
from pathlib import Path


# do not remove - hides the libraries' debug messages
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)


SYMAI_VERSION = "0.2.27"
__version__   = SYMAI_VERSION
__root_dir__  = Path.home() / '.symai'


def _start_symai():
    global _ai_config_

    # CREATE THE SYMAI FOLDER IF IT DOES NOT EXIST YET
    # *==============================================================================================================*
    if not os.path.exists(__root_dir__):
        os.makedirs(__root_dir__)

    # CHECK IF THE USER HAS A CONFIGURATION FILE IN THE CURRENT WORKING DIRECTORY (DEBUGGING MODE)
    # *==============================================================================================================*
    if os.path.exists(Path.cwd() / 'symai.config.json'):
        logging.warn('Using the configuration file in the current working directory.')
        _ai_config_path_ = Path.cwd() / 'symai.config.json'

    else:
        # CREATE THE CONFIGURATION FILE IF IT DOES NOT EXIST YET WITH THE DEFAULT VALUES
        # *==========================================================================================================*
        _ai_config_path_ = __root_dir__ / 'symai.config.json'

        if not os.path.exists(_ai_config_path_):
            with open(_ai_config_path_, 'w') as f:
                json.dump({
                    "NEUROSYMBOLIC_ENGINE_API_KEY":   "",
                    "NEUROSYMBOLIC_ENGINE_MODEL":     "text-davinci-003",
                    "SYMBOLIC_ENGINE_API_KEY":        "",
                    "SYMBOLIC_ENGINE":                "wolframalpha",
                    "EMBEDDING_ENGINE_API_KEY":       "",
                    "EMBEDDING_ENGINE_MODEL":         "text-embedding-ada-002",
                    "IMAGERENDERING_ENGINE_API_KEY":  "",
                    "VISION_ENGINE_MODEL":            "openai/clip-vit-base-patch32",
                    "SEARCH_ENGINE_API_KEY":          "",
                    "SEARCH_ENGINE_MODEL":            "google",
                    "OCR_ENGINE_API_KEY":             "",
                    "SPEECH_ENGINE_MODEL":            "base",
                    "INDEXING_ENGINE_API_KEY":        "",
                    "INDEXING_ENGINE_ENVIRONMENT":    "us-west1-gcp"
                }, f, indent=4)

        # LOAD THE CONFIGURATION FILE
        # *==========================================================================================================*
        with open(_ai_config_path_, 'r') as f:
            _ai_config_ = json.load(f)
        _tmp_ai_config_ = _ai_config_.copy()

        # LOAD THE ENVIRONMENT VARIABLES
        # *==========================================================================================================*
        _openai_api_key_                = os.environ.get('OPENAI_API_KEY', None)
        _neurosymbolic_engine_api_key_  = os.environ.get('NEUROSYMBOLIC_ENGINE_API_KEY', None)
        _neurosymbolic_engine_model_    = os.environ.get('NEUROSYMBOLIC_ENGINE_MODEL', None)
        _symbolic_engine_api_key_       = os.environ.get('SYMBOLIC_ENGINE_API_KEY', None)
        _embedding_engine_api_key_      = os.environ.get('EMBEDDING_ENGINE_API_KEY', None)
        _embedding_engine_model_        = os.environ.get('EMBEDDING_ENGINE_MODEL', None)
        _imagerendering_engine_api_key_ = os.environ.get('IMAGERENDERING_ENGINE_API_KEY', None)
        _vision_engine_model_           = os.environ.get('VISION_ENGINE_MODEL', None)
        _search_engine_api_key_         = os.environ.get('SEARCH_ENGINE_API_KEY', None)
        _search_engine_model_           = os.environ.get('SEARCH_ENGINE_MODEL', None)
        _ocr_engine_api_key_            = os.environ.get('OCR_ENGINE_API_KEY', None)
        _speech_engine_model_           = os.environ.get('SPEECH_ENGINE_MODEL', None)
        _indexing_engine_api_key_       = os.environ.get('INDEXING_ENGINE_API_KEY', None)
        _indexing_engine_environment_   = os.environ.get('INDEXING_ENGINE_ENVIRONMENT', None)

        # SET/UPDATE THE API KEYS
        # *==========================================================================================================*
        if _neurosymbolic_engine_api_key_:  _ai_config_['NEUROSYMBOLIC_ENGINE_API_KEY']  = _neurosymbolic_engine_api_key_
        if _symbolic_engine_api_key_:       _ai_config_['SYMBOLIC_ENGINE_API_KEY']       = _symbolic_engine_api_key_
        if _embedding_engine_api_key_:      _ai_config_['EMBEDDING_ENGINE_API_KEY']      = _embedding_engine_api_key_
        if _imagerendering_engine_api_key_: _ai_config_['IMAGERENDERING_ENGINE_API_KEY'] = _imagerendering_engine_api_key_

        # USE ENVIRONMENT VARIABLES IF THE USER DID NOT SET THE API KEYS
        # *==========================================================================================================*
        if _openai_api_key_ and not _neurosymbolic_engine_api_key_:  _ai_config_['NEUROSYMBOLIC_ENGINE_API_KEY']  = _openai_api_key_
        if _openai_api_key_ and not _embedding_engine_api_key_:      _ai_config_['EMBEDDING_ENGINE_API_KEY']      = _openai_api_key_
        if _openai_api_key_ and not _imagerendering_engine_api_key_: _ai_config_['IMAGERENDERING_ENGINE_API_KEY'] = _openai_api_key_

        # OPTIONAL MODULES
        # *==========================================================================================================*
        if _neurosymbolic_engine_model_:  _ai_config_['NEUROSYMBOLIC_ENGINE_MODEL']  = _neurosymbolic_engine_model_
        if _embedding_engine_model_:      _ai_config_['EMBEDDING_ENGINE_MODEL']      = _embedding_engine_model_
        if _vision_engine_model_:         _ai_config_['VISION_ENGINE_MODEL']         = _vision_engine_model_
        if _search_engine_api_key_:       _ai_config_['SEARCH_ENGINE_API_KEY']       = _search_engine_api_key_
        if _search_engine_model_:         _ai_config_['SEARCH_ENGINE_MODEL']         = _search_engine_model_
        if _ocr_engine_api_key_:          _ai_config_['OCR_ENGINE_API_KEY']          = _ocr_engine_api_key_
        if _speech_engine_model_:         _ai_config_['SPEECH_ENGINE_MODEL']         = _speech_engine_model_
        if _indexing_engine_api_key_:     _ai_config_['INDEXING_ENGINE_API_KEY']     = _indexing_engine_api_key_
        if _indexing_engine_environment_: _ai_config_['INDEXING_ENGINE_ENVIRONMENT'] = _indexing_engine_environment_

        # VERIFY IF THE CONFIGURATION FILE HAS CHANGED AND UPDATE IT
        # *==========================================================================================================*
        _updated_key_ = {k: not _tmp_ai_config_[k] == _ai_config_[k] for k in _tmp_ai_config_.keys()}
        _has_changed_ = any(_updated_key_.values())

        if _has_changed_:
            # update the symai.config.json file
            with open(_ai_config_path_, 'w') as f:
                json.dump(_ai_config_, f, indent=4)

    # CHECK IF MANADATORY API KEYS ARE SET
    # *==============================================================================================================*
    with open(_ai_config_path_, 'r') as f:
        _ai_config_ = json.load(f)

    if 'custom' not in _ai_config_['NEUROSYMBOLIC_ENGINE_MODEL'].lower() and \
                      (_ai_config_['NEUROSYMBOLIC_ENGINE_API_KEY'] is None or len(_ai_config_['NEUROSYMBOLIC_ENGINE_API_KEY']) == 0):

        logging.warn('The mandatory neuro-symbolic engine is not initialized. Please get a key from https://beta.openai.com/account/api-keys and set either a general environment variable OPENAI_API_KEY or a module specific environment variable NEUROSYMBOLIC_ENGINE_API_KEY.')

    import symai.backend.settings as settings
    settings.SYMAI_CONFIG = _ai_config_


_start_symai()


from .backend.base import Engine
from .chat import ChatBot
from .components import *
from .core import *
from .functional import ConstraintViolationException
from .memory import *
from .post_processors import *
from .symbol import *
from .interfaces import *
from .pre_processors import *
from .prompts import Prompt
from .shell import Shell
from .symbol import *
