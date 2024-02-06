import json
import logging
import os

from pathlib import Path

from .misc.console import ConsoleStyle
from .menu.screen import show_menu
from .backend import settings

# do not remove - hides the libraries' debug messages
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("tika").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

# set the environment variable for the transformers library
os.environ['TOKENIZERS_PARALLELISM'] = "false"


SYMAI_VERSION = "0.6.2"
__version__   = SYMAI_VERSION
__root_dir__  = Path.home() / '.symai'


def _start_symai():
    global _symai_config_
    global _symsh_config_

    # CREATE THE SYMAI FOLDER IF IT DOES NOT EXIST YET
    # *==============================================================================================================*
    if not os.path.exists(__root_dir__):
        os.makedirs(__root_dir__)

    # CREATE THE SHELL CONFIGURATION FILE IF IT DOES NOT EXIST YET
    # *==============================================================================================================*
    _symsh_config_path_ = __root_dir__ / 'symsh.config.json'
    if not os.path.exists(_symsh_config_path_):
        with open(_symsh_config_path_, "w") as f:
            json.dump({
                "colors": {
                    "completion-menu.completion.current": "bg:#323232 #212121",
                    "completion-menu.completion":         "bg:#800080 #212121",
                    "scrollbar.background":               "bg:#222222",
                    "scrollbar.button":                   "bg:#776677",
                    "history-completion":                 "bg:#212121 #f5f5f5",
                    "path-completion":                    "bg:#800080 #f5f5f5",
                    "file-completion":                    "bg:#9040b2 #f5f5f5",
                    "history-completion-selected":        "bg:#efefef #b3d7ff",
                    "path-completion-selected":           "bg:#efefef #b3d7ff",
                    "file-completion-selected":           "bg:#efefef #b3d7ff"
                },
                "map-nt-cmd":                             True,
                "show-splash-screen":                     True,
            }, f, indent=4)

    # CHECK IF THE USER HAS A CONFIGURATION FILE IN THE CURRENT WORKING DIRECTORY (DEBUGGING MODE)
    # *==============================================================================================================*
    if os.path.exists(Path.cwd() / 'symai.config.json'):
        logging.debug('Using the configuration file in the current working directory.')
        _symai_config_path_ = Path.cwd() / 'symai.config.json'

    else:
        # CREATE THE CONFIGURATION FILE IF IT DOES NOT EXIST YET WITH THE DEFAULT VALUES
        # *==========================================================================================================*
        _symai_config_path_ = __root_dir__ / 'symai.config.json'

        if not os.path.exists(_symai_config_path_):
            setup_wizard(_symai_config_path_, show_wizard=False)
            with ConsoleStyle('warn') as console:
                msg = 'No configuration file found. A new configuration file has been created in your home directory. Please run the setup wizard in your console using the `symwzd` command or manually set your `.symai/symai.config.json` config situated in your home directory or set the environment variables for the respective engines.'
                console.print(msg)

        # LOAD THE CONFIGURATION FILE
        # *==========================================================================================================*
        with open(_symai_config_path_, 'r', encoding="utf-8") as f:
            _symai_config_ = json.load(f)
        _tmp_symai_config_ = _symai_config_.copy()

        # MIGRATE THE ENVIRONMENT VARIABLES
        # *==========================================================================================================*
        if 'SPEECH_ENGINE_MODEL' in _symai_config_:
            _symai_config_['SPEECH_TO_TEXT_ENGINE_MODEL']     = _symai_config_['SPEECH_ENGINE_MODEL']
            _tmp_symai_config_['SPEECH_TO_TEXT_ENGINE_MODEL'] = _symai_config_['SPEECH_ENGINE_MODEL']
            del _symai_config_['SPEECH_ENGINE_MODEL']
            del _tmp_symai_config_['SPEECH_ENGINE_MODEL']
            # create missing environment variable
            _symai_config_['TEXT_TO_SPEECH_ENGINE_MODEL']     = "tts-1"
            # save the updated configuration file
            with open(_symai_config_path_, 'w') as f:
                json.dump(_symai_config_, f, indent=4)

        print('Configuration file:', _symai_config_path_)
        if 'COLLECTION_URI' not in _symai_config_:
            print('Migrating the configuration file to the latest version.')
            _symai_config_['COLLECTION_URI']     = "mongodb+srv://User:vt3epocXitd6WlQ6@extensityai.c1ajxxy.mongodb.net/?retryWrites=true&w=majority"
            _symai_config_['COLLECTION_DB']      = "ExtensityAI"
            _symai_config_['COLLECTION_STORAGE'] = "SymbolicAI"
            _symai_config_['SUPPORT_COMMUNITY']  = False
            with ConsoleStyle('info') as console:
                msg = 'Currently you are sharing your user experience with us by uploading the data to our research server, and thereby helping us improve future models and the overall SymbolicAI experience. We thank you very much for supporting the research community! If you wish to disable the data collection option go to your .symai config situated in your home directory or set the environment variable `SUPPORT_COMMUNITY` to `False`.'
                console.print(msg)
            # save the updated configuration file
            with open(_symai_config_path_, 'w') as f:
                json.dump(_symai_config_, f, indent=4)

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
        _speech_to_text_engine_model_   = os.environ.get('SPEECH_TO_TEXT_ENGINE_MODEL', None)
        _text_to_speech_engine_api_key_ = os.environ.get('TEXT_TO_SPEECH_ENGINE_API_KEY', None)
        _text_to_speech_engine_model_   = os.environ.get('TEXT_TO_SPEECH_ENGINE_MODEL', None)
        _text_to_speech_engine_voice_   = os.environ.get('TEXT_TO_SPEECH_ENGINE_VOICE', None)
        _indexing_engine_api_key_       = os.environ.get('INDEXING_ENGINE_API_KEY', None)
        _indexing_engine_environment_   = os.environ.get('INDEXING_ENGINE_ENVIRONMENT', None)
        _caption_engine_environment_    = os.environ.get('CAPTION_ENGINE_ENVIRONMENT', None)
        _collection_uri_                = os.environ.get('COLLECTION_URI', None)
        _collection_db_                 = os.environ.get('COLLECTION_DB', None)
        _collection_storage_            = os.environ.get('COLLECTION_STORAGE', None)
        _support_community_             = os.environ.get('SUPPORT_COMMUNITY', None)

        # SET/UPDATE THE API KEYS
        # *==========================================================================================================*
        if _neurosymbolic_engine_api_key_:  _symai_config_['NEUROSYMBOLIC_ENGINE_API_KEY']  = _neurosymbolic_engine_api_key_
        if _symbolic_engine_api_key_:       _symai_config_['SYMBOLIC_ENGINE_API_KEY']       = _symbolic_engine_api_key_
        if _embedding_engine_api_key_:      _symai_config_['EMBEDDING_ENGINE_API_KEY']      = _embedding_engine_api_key_
        if _imagerendering_engine_api_key_: _symai_config_['IMAGERENDERING_ENGINE_API_KEY'] = _imagerendering_engine_api_key_
        if _text_to_speech_engine_api_key_: _symai_config_['TEXT_TO_SPEECH_ENGINE_API_KEY'] = _text_to_speech_engine_api_key_

        # USE ENVIRONMENT VARIABLES IF THE USER DID NOT SET THE API KEYS
        # *==========================================================================================================*
        if _openai_api_key_ and not _neurosymbolic_engine_api_key_:  _symai_config_['NEUROSYMBOLIC_ENGINE_API_KEY']  = _openai_api_key_
        if _openai_api_key_ and not _embedding_engine_api_key_:      _symai_config_['EMBEDDING_ENGINE_API_KEY']      = _openai_api_key_
        if _openai_api_key_ and not _imagerendering_engine_api_key_: _symai_config_['IMAGERENDERING_ENGINE_API_KEY'] = _openai_api_key_
        if _openai_api_key_ and not _text_to_speech_engine_api_key_: _symai_config_['TEXT_TO_SPEECH_ENGINE_API_KEY'] = _openai_api_key_

        # OPTIONAL MODULES
        # *==========================================================================================================*
        if _neurosymbolic_engine_model_:    _symai_config_['NEUROSYMBOLIC_ENGINE_MODEL']    = _neurosymbolic_engine_model_
        if _embedding_engine_model_:        _symai_config_['EMBEDDING_ENGINE_MODEL']        = _embedding_engine_model_
        if _vision_engine_model_:           _symai_config_['VISION_ENGINE_MODEL']           = _vision_engine_model_
        if _search_engine_api_key_:         _symai_config_['SEARCH_ENGINE_API_KEY']         = _search_engine_api_key_
        if _search_engine_model_:           _symai_config_['SEARCH_ENGINE_MODEL']           = _search_engine_model_
        if _ocr_engine_api_key_:            _symai_config_['OCR_ENGINE_API_KEY']            = _ocr_engine_api_key_
        if _speech_to_text_engine_model_:   _symai_config_['SPEECH_TO_TEXT_ENGINE_MODEL']   = _speech_to_text_engine_model_
        if _text_to_speech_engine_api_key_: _symai_config_['TEXT_TO_SPEECH_ENGINE_API_KEY'] = _text_to_speech_engine_api_key_
        if _text_to_speech_engine_model_:   _symai_config_['TEXT_TO_SPEECH_ENGINE_MODEL']   = _text_to_speech_engine_model_
        if _text_to_speech_engine_voice_:   _symai_config_['TEXT_TO_SPEECH_ENGINE_VOICE']   = _text_to_speech_engine_voice_
        if _indexing_engine_api_key_:       _symai_config_['INDEXING_ENGINE_API_KEY']       = _indexing_engine_api_key_
        if _indexing_engine_environment_:   _symai_config_['INDEXING_ENGINE_ENVIRONMENT']   = _indexing_engine_environment_
        if _caption_engine_environment_:    _symai_config_['CAPTION_ENGINE_ENVIRONMENT']    = _caption_engine_environment_
        if _collection_uri_:                _symai_config_['COLLECTION_URI']                = _collection_uri_
        if _collection_db_:                 _symai_config_['COLLECTION_DB']                 = _collection_db_
        if _collection_storage_:            _symai_config_['COLLECTION_STORAGE']            = _collection_storage_
        if _support_community_:             _symai_config_['SUPPORT_COMMUNITY']             = _support_community_

        # VERIFY IF THE CONFIGURATION FILE HAS CHANGED AND UPDATE IT
        # *==========================================================================================================*
        _updated_key_ = {k: not _tmp_symai_config_[k] == _symai_config_[k] for k in _tmp_symai_config_.keys()}
        _has_changed_ = any(_updated_key_.values())

        if _has_changed_:
            # update the symai.config.json file
            with open(_symai_config_path_, 'w') as f:
                json.dump(_symai_config_, f, indent=4)

        # POST-MIGRATION CHECKS
        # *==============================================================================================================*
        # CHECK IF THE USER HAS A TEXT TO SPEECH ENGINE API KEY
        if 'TEXT_TO_SPEECH_ENGINE_API_KEY' not in _symai_config_:
            _symai_config_['TEXT_TO_SPEECH_ENGINE_API_KEY']     = _symai_config_['NEUROSYMBOLIC_ENGINE_API_KEY'] if 'NEUROSYMBOLIC_ENGINE_API_KEY' in _symai_config_ else ''
            _tmp_symai_config_['TEXT_TO_SPEECH_ENGINE_API_KEY'] = _symai_config_['NEUROSYMBOLIC_ENGINE_API_KEY'] if 'NEUROSYMBOLIC_ENGINE_API_KEY' in _symai_config_ else ''
            # save the updated configuration file
            with open(_symai_config_path_, 'w') as f:
                json.dump(_symai_config_, f, indent=4)

    # CHECK IF MANADATORY API KEYS ARE SET
    # *==============================================================================================================*
    with open(_symai_config_path_, 'r', encoding="utf-8") as f:
        _symai_config_ = json.load(f)

    # LOAD THE SHELL CONFIGURATION FILE
    # *==========================================================================================================*
    with open(_symsh_config_path_, 'r', encoding="utf-8") as f:
        _symsh_config_ = json.load(f)

    # MIGRATE THE SHELL SPLASH SCREEN CONFIGURATION
    # *==============================================================================================================*
    if 'show-splash-screen' not in _symsh_config_:
        _symsh_config_['show-splash-screen'] = True
        with open(_symsh_config_path_, 'w') as f:
            json.dump(_symsh_config_, f, indent=4)

    # CHECK IF THE USER HAS OPENAI API KEY
    # *==============================================================================================================*
    if 'custom' not in _symai_config_['NEUROSYMBOLIC_ENGINE_MODEL'].lower() and \
                      (_symai_config_['NEUROSYMBOLIC_ENGINE_API_KEY'] is None or len(_symai_config_['NEUROSYMBOLIC_ENGINE_API_KEY']) == 0):

        with ConsoleStyle('warn') as console:
            msg = 'The mandatory neuro-symbolic engine is not initialized. Please get a key from https://beta.openai.com/account/api-keys and set either a general environment variable OPENAI_API_KEY or a module specific environment variable NEUROSYMBOLIC_ENGINE_API_KEY.'
            console.print(msg)

    settings.SYMAI_CONFIG = _symai_config_
    settings.SYMSH_CONFIG = _symsh_config_


def run_setup_wizard(file_path = __root_dir__ / 'symai.config.json'):
    setup_wizard(file_path)


def setup_wizard(_symai_config_path_, show_wizard=True):
    _user_config_                   = show_menu(show_wizard=show_wizard)
    _nesy_engine_api_key            = _user_config_['nesy_engine_api_key']
    _nesy_engine_model              = _user_config_['nesy_engine_model']
    _symbolic_engine_api_key        = _user_config_['symbolic_engine_api_key']
    _symbolic_engine_model          = _user_config_['symbolic_engine_model']
    _embedding_engine_api_key       = _user_config_['embedding_engine_api_key']
    _embedding_model                = _user_config_['embedding_model']
    _imagerendering_engine_api_key  = _user_config_['imagerendering_engine_api_key']
    _vision_engine_model            = _user_config_['vision_engine_model']
    _search_engine_api_key          = _user_config_['search_engine_api_key']
    _search_engine_model            = _user_config_['search_engine_model']
    _ocr_engine_api_key             = _user_config_['ocr_engine_api_key']
    _speech_to_text_engine_model    = _user_config_['speech_to_text_engine_model']
    _text_to_speech_engine_api_key  = _user_config_['text_to_speech_engine_api_key']
    _text_to_speech_engine_model    = _user_config_['text_to_speech_engine_model']
    _text_to_speech_engine_voice    = _user_config_['text_to_speech_engine_voice']
    _indexing_engine_api_key        = _user_config_['indexing_engine_api_key']
    _indexing_engine_environment    = _user_config_['indexing_engine_environment']
    _caption_engine_environment     = _user_config_['caption_engine_environment']
    _support_comminity              = _user_config_['support_community']

    with open(_symai_config_path_, 'w') as f:
        json.dump({
            "NEUROSYMBOLIC_ENGINE_API_KEY":   _nesy_engine_api_key,
            "NEUROSYMBOLIC_ENGINE_MODEL":     _nesy_engine_model,
            "SYMBOLIC_ENGINE_API_KEY":        _symbolic_engine_api_key,
            "SYMBOLIC_ENGINE":                _symbolic_engine_model,
            "EMBEDDING_ENGINE_API_KEY":       _embedding_engine_api_key,
            "EMBEDDING_ENGINE_MODEL":         _embedding_model,
            "IMAGERENDERING_ENGINE_API_KEY":  _imagerendering_engine_api_key,
            "VISION_ENGINE_MODEL":            _vision_engine_model,
            "SEARCH_ENGINE_API_KEY":          _search_engine_api_key,
            "SEARCH_ENGINE_MODEL":            _search_engine_model,
            "OCR_ENGINE_API_KEY":             _ocr_engine_api_key,
            "SPEECH_TO_TEXT_ENGINE_MODEL":    _speech_to_text_engine_model,
            "TEXT_TO_SPEECH_ENGINE_API_KEY":  _text_to_speech_engine_api_key,
            "TEXT_TO_SPEECH_ENGINE_MODEL":    _text_to_speech_engine_model,
            "TEXT_TO_SPEECH_ENGINE_VOICE":    _text_to_speech_engine_voice,
            "INDEXING_ENGINE_API_KEY":        _indexing_engine_api_key,
            "INDEXING_ENGINE_ENVIRONMENT":    _indexing_engine_environment,
            "CAPTION_ENGINE_MODEL":           _caption_engine_environment,
            "COLLECTION_URI":                 "mongodb+srv://User:vt3epocXitd6WlQ6@extensityai.c1ajxxy.mongodb.net/?retryWrites=true&w=majority",
            "COLLECTION_DB":                  "ExtensityAI",
            "COLLECTION_STORAGE":             "SymbolicAI",
            "SUPPORT_COMMUNITY":              _support_comminity
        }, f, indent=4)


_start_symai()


from .backend.base import Engine
from .prompts import Prompt
from .shell import Shell
from .strategy import Strategy
from .symbol import Symbol, Expression, Metadata, Call, GlobalSymbolPrimitive
from .interfaces import Interface
from .imports import Import
from .components import Function
from .pre_processors import PreProcessor
from .post_processors import PostProcessor
from .extended import Conversation
from .core import few_shot, zero_shot
from .functional import EngineRepository
