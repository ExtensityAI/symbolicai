import json
import os
from box import Box
from rpyc.utils.server import ThreadedServer

# check if huggingface mdoel is already initialized
_hf_config_path_ = os.path.join(os.getcwd(), 'huggingface_causallm.config.json')
if not os.path.exists(_hf_config_path_):
    _parent_dir_ = os.path.dirname(__file__)
    _hf_config_path_ = os.path.join(_parent_dir_, 'backend', 'services', 'configs', 'huggingface_causallm.config.json')
# read the huggingface.config.json file
with open(_hf_config_path_, 'r') as f:
    _args_ = Box(json.load(f))
os.makedirs(_args_.huggingface_cache, exist_ok=True)


def run() -> None:
    from symai.backend.services.huggingface_causallm_server import HuggingFaceService
    server = ThreadedServer(HuggingFaceService, port=_args_.port)
    server.start()
