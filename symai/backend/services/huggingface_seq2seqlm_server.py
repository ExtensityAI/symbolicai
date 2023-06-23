import json
import logging
import os
from abc import ABC

import rpyc
import torch
from box import Box
from rpyc.utils.server import ThreadedServer
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList, set_seed)

# check if huggingface mdoel is already initialized
_hf_config_path_ = os.path.join(os.getcwd(), 'huggingface_seq2seqlm.config.json')
if not os.path.exists(_hf_config_path_):
    _parent_dir_ = os.path.dirname(__file__)
    _hf_config_path_ = os.path.join(_parent_dir_, 'configs', 'huggingface_seq2seqlm.config.json')
# read the huggingface.config.json file
with open(_hf_config_path_, 'r') as f:
    _args_ = Box(json.load(f))


class MaxSentenceStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list, max_sentences=5):
        self.args = _args_
        self.keywords = keywords_ids
        self.count = 0
        self.max_sentences = max_sentences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in set(self.keywords):
            if self.count >= self.max_sentences:
                return True
            else:
                self.count += 1
                return False
        return False


class HuggingFaceModel(ABC):
    def __init__(self):
        super().__init__()
        
        self.logger = logging.getLogger('huggingface')
        self.logger.setLevel(logging.WARNING)
        self.args = _args_
        self.model = None
        self.tokenizer = None
        
        msg = f'Starting HuggingFace model service on port: {_args_.port}'
        self.logger.info(msg)
        print(msg)        
    
    def init_model(self, device = None):
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
        
        model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model, 
                                                      torch_dtype=dtype, 
                                                      pad_token_id=tokenizer.eos_token_id)

        model.to(device)
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        msg = f'Model initialization finished! Ready to play...'
        self.logger.info(msg)
        print(msg)   
    
    def predict(self, input, 
                max_tokens: int,
                temperature: float,
                top_p: float,
                top_k: int,
                stop: str = None, 
                suffix: str = '', # TODO: think of proper equivalent to OPENAI's suffix
                do_sample=True):
        top_k = top_k if do_sample else 0
        top_p = top_p if do_sample else 0
        
        text = input

        input_ids = self.tokenizer(text).input_ids
        input_ids = torch.Tensor(input_ids).long().to(self.model.device)
        
        stopping_criteria = None
        if stop is not None:
            stop_words = [stop]
            stop_ids = [torch.Tensor(self.tokenizer(w).input_ids[0]).long().to(self.model.device) for w in stop_words]
            stop_criteria1 = MaxSentenceStoppingCriteria(stop_ids, max_sentences=3)
            stopping_criteria = StoppingCriteriaList([stop_criteria1])
        
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        
        with torch.no_grad():
            generated_ids = self.model.generate(input_ids,
                                                # fixed settings
                                                num_return_sequences=1,
                                                remove_invalid_values=True,
                                                # configurable settings
                                                max_new_tokens=max_tokens,
                                                temperature=temperature,
                                                num_beams=self.args.num_beams,
                                                no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                                                # program logic setting
                                                do_sample=do_sample,
                                                top_k=top_k,
                                                top_p=top_p,
                                                force_words_ids=None,
                                                stopping_criteria=stopping_criteria)
        
        rsp = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # remove the input text from the response
        # TODO: this is a hack, need to find a better way to do this
        if '=>' in text:
            res = rsp.split('------------')[-1]
            res = res.split('=>')[-1]
            rsp = res.strip()
        elif ':' in text:
            rsp = rsp.split(':')[-1].strip()
        else:
            rsp = rsp.replace(text, '').strip()
        return rsp


_model_ = None


@rpyc.service
class HuggingFaceService(rpyc.Service):
    
    @rpyc.exposed
    def predict(self, *args, **kwargs):
        global _model_
        if _model_ is None:
            _model_ = HuggingFaceModel()
            _model_.init_model(device=_args_.device)
        if len(args) == 0:
            return '<startup_sequence/>'
        return _model_.predict(*args, **kwargs)


if __name__ == '__main__':
    server = ThreadedServer(HuggingFaceService, port=_args_.port)
    server.start()
