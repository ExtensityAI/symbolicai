import json
import os
import rpyc

from box import Box

rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True


# check if huggingface mdoel is already initialized
_hf_config_path_ = os.path.join(os.getcwd(), 'huggingface_causallm.config.json')
if not os.path.exists(_hf_config_path_):
    _parent_dir_ = os.path.dirname(__file__)
    _hf_config_path_ = os.path.join(_parent_dir_, 'backend', 'services', 'configs', 'huggingface_causallm.config.json')
# read the huggingface.config.json file
with open(_hf_config_path_, 'r') as f:
    _args_ = Box(json.load(f))


def run() -> None:
    connection = rpyc.connect('localhost', _args_.port)
    connection._config['sync_request_timeout'] = 1000
    input_ = 'Return one number between 1 and 10: x ='
    print('input:', input_)
    print('max_tokens:', connection.root.max_tokens())
    res = connection.root.predict(input_, max_new_tokens=30, do_sample=True, top_k=10, top_p=0.95, temperature=0.7, num_return_sequences=1, stop_words=['\n'])
    print('res:', res)
