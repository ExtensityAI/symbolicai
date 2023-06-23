import json
import os

import rpyc
from box import Box

# check if huggingface mdoel is already initialized
_hf_config_path_ = os.path.join(os.getcwd(), 'huggingface_causallm.config.json')
if not os.path.exists(_hf_config_path_):
    _parent_dir_ = os.path.dirname(__file__)
    _hf_config_path_ = os.path.join(_parent_dir_, 'configs', 'huggingface_causallm.config.json')
# read the huggingface.config.json file
with open(_hf_config_path_, 'r') as f:
    _args_ = Box(json.load(f))


if __name__ == '__main__':
    connection = rpyc.connect('localhost', _args_.port)
    connection._config['sync_request_timeout'] = 1000
    connection.root.predict()
