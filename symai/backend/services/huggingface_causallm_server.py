import json
import logging
import os
from abc import ABC
from pathlib import Path
from typing import List

import rpyc
import torch
from box import Box
from rpyc.utils.server import ThreadedServer

rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True


# check if huggingface mdoel is already initialized
_hf_config_path_ = os.path.join(os.getcwd(), 'huggingface_causallm.config.json')
if not os.path.exists(_hf_config_path_):
    _parent_dir_ = os.path.dirname(__file__)
    _hf_config_path_ = os.path.join(_parent_dir_, 'configs', 'huggingface_causallm.config.json')
# read the huggingface.config.json file
with open(_hf_config_path_, 'r') as f:
    _args_ = Box(json.load(f))
_args_.huggingface_cache  = str(Path.home() / _args_.huggingface_cache)
print('huggingface_cache:', _args_.huggingface_cache)
os.environ['TRANSFORMERS_CACHE'] = _args_.huggingface_cache # set the environment variable for transformers cache


from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig, StoppingCriteria,
                          StoppingCriteriaList, set_seed)


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], device='cpu'):
        super().__init__()
        self.stops = [stop.to(device) for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.any((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


class HuggingFaceModel(ABC):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger('huggingface')
        self.logger.setLevel(logging.WARNING)
        self.args      = _args_
        self.model     = None
        self.tokenizer = None

    def init_model(self, device = None, verbose = False):
        msg = f'Initializing HuggingFace model weights: {_args_.model}'
        self.logger.info(msg)
        print(msg)

        try:
            if device is not None:
                device = torch.device(device)
        except:
            pass
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.args.seed is not None:
            set_seed(self.args.seed)
        dtype = torch.float16 if self.args.dtype == 'float16' else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(self.args.model,
                                                  use_fast=False,
                                                  local_files_only=False,
                                                  output_hidden_states=True)

        model = AutoModelForCausalLM.from_pretrained(self.args.model,
                                                     torch_dtype=dtype,
                                                     low_cpu_mem_usage=True,
                                                     local_files_only=False,
                                                     pad_token_id=tokenizer.eos_token_id)

        generation_config = GenerationConfig.from_pretrained(self.args.model)

        model.to(device)
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.verbose = verbose
        msg = f'Model initialization finished! Ready to play...'
        self.logger.info(msg)
        print(msg)

    def max_tokens(self, *args, **kwargs):
        return self.args.max_seq_len

    def predict(self, input, **kwargs):
        if input is None:
            return None
        text = input.strip()
        if self.verbose:
            print('input:', text)

        input_ids = self.tokenizer(text).input_ids
        input_ids = torch.Tensor(input_ids).long().to(self.model.device)

        if hasattr(kwargs, 'stopping_criteria'):
            kwargs['stopping_criteria'] = self.generation_config.stopping_criteria
        else:
            stop_words                  = ['\n', '\r'] if 'stop_words' not in kwargs else kwargs['stop_words']
            stop_words_ids              = [self.tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]
            stopping_criteria           = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids, device=self.model.device)])
            kwargs['early_stopping']    = True
            kwargs['stopping_criteria'] = stopping_criteria

        if self.verbose:
            print('kwargs:', kwargs)

        generation_config = GenerationConfig(
            **self.generation_config.to_dict(),
        )

        # override configs based on kwargs
        for k, v in kwargs.items():
            setattr(generation_config, k, v)

        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)

        with torch.no_grad():
            config = generation_config.to_dict()
            if 'max_length' in config: # recommended to remove by HuggingFace
                del config['max_length']
            if 'stop_words' in config: # clean up since not supported by HuggingFace
                del config['stop_words']
            if 'suffix' in config: # clean up since not supported by HuggingFace
                del config['suffix']
            if 'max_tokens' in config: # rename to max_new_tokens
                config['max_new_tokens'] = config['max_tokens']
                del config['max_tokens']
            generated_ids = self.model.generate(input_ids, **config)

        rsp = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        if self.verbose:
            print('rsp:', rsp)
        # remove the input text from the response
        rsp = rsp.replace(f'<s>{text}', '').strip()
        return rsp


_model_ = None


@rpyc.service
class HuggingFaceService(rpyc.Service):

    @staticmethod
    def init_model():
        global _model_
        if _model_ is None:
            _model_ = HuggingFaceModel()
            _model_.init_model(device=_args_.device, verbose=_args_.verbose)

    @rpyc.exposed
    def predict(self, *args, **kwargs):
        if len(args) == 0:
            return '<startup_sequence/>'
        return _model_.predict(*args, **kwargs)

    @rpyc.exposed
    def max_tokens(self, *args, **kwargs):
        return _model_.max_tokens(*args, **kwargs)


if __name__ == '__main__':
    msg = f'Starting HuggingFace model service on port: {_args_.port}'
    print(msg)
    HuggingFaceService.init_model()
    server = ThreadedServer(HuggingFaceService, port=_args_.port)
    server.start()
