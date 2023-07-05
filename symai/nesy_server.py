import json
import os
from pathlib import Path

from box import Box
from rpyc.utils.server import ThreadedServer

from .backend.services.huggingface_causallm_server import HuggingFaceService

# check if huggingface mdoel is already initialized
_hf_config_path_ = os.path.join(os.getcwd(), 'huggingface_causallm.config.json')
if not os.path.exists(_hf_config_path_):
    _parent_dir_ = os.path.dirname(__file__)
    _hf_config_path_ = os.path.join(_parent_dir_, 'backend', 'services', 'configs', 'huggingface_causallm.config.json')
# read the huggingface.config.json file
with open(_hf_config_path_, 'r') as f:
    _args_ = Box(json.load(f))
_args_.huggingface_cache = __root_dir__  = str(Path.home() / _args_.huggingface_cache)
os.makedirs(__root_dir__, exist_ok=True)


def run() -> None:
    msg = f'Starting model service on port: {_args_.port}'
    print(msg)
    HuggingFaceService.init_model()
    server = ThreadedServer(HuggingFaceService, port=_args_.port)
    server.start()
