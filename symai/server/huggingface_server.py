import argparse
import random
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, StoppingCriteria,
                          StoppingCriteriaList)

# General arguments
parser = argparse.ArgumentParser(description="FastAPI server for Hugging Face models")
parser.add_argument("--model", type=str, help="Path to the model")
parser.add_argument("--host", type=str, default="localhost", help="Host address. (default: localhost)")
parser.add_argument("--port", type=int, default=8000, help="Port number. (default: 8000)")

# Quantization arguments with 'quant_' prefix
parser.add_argument("--quant", action="store_true", default=False, help="Enable quantization; see help for available quantization options (default: False)")
parser.add_argument("--quant_load_in_8bit", action="store_true", default=False, help="Load model in 8-bit precision (default: False)")
parser.add_argument("--quant_load_in_4bit", action="store_true", default=False, help="Load model in 4-bit precision (default: False)")
parser.add_argument("--quant_llm_int8_threshold", type=float, default=6.0, help="LLM int8 threshold (default: 6.0)")
parser.add_argument("--quant_llm_int8_skip_modules", type=str, nargs="+", default=None, help="LLM int8 skip modules (default: None)")
parser.add_argument("--quant_llm_int8_enable_fp32_cpu_offload", action="store_true", default=False, help="Enable FP32 CPU offload for LLM int8 (default: False)")
parser.add_argument("--quant_llm_int8_has_fp16_weight", action="store_true", default=False, help="LLM int8 has FP16 weight (default: False)")
parser.add_argument("--quant_bnb_4bit_compute_dtype", type=str, default=None, help="BNB 4-bit compute dtype (default: None)")
parser.add_argument("--quant_bnb_4bit_quant_type", type=str, default="fp4", help="BNB 4-bit quantization type (default: fp4)")
parser.add_argument("--quant_bnb_4bit_use_double_quant", action="store_true", default=False, help="Use double quantization for BNB 4-bit (default: False)")
parser.add_argument("--quant_bnb_4bit_quant_storage", type=str, default=None, help="BNB 4-bit quantization storage (default: None)")

# Model inference arguments
# https://huggingface.co/docs/transformers/main/en/main_classes/model
parser.add_argument("--torch_dtype", type=str, default="auto", help="A string that is a valid torch.dtype. E.g. “float32” loads the model in torch.float32, “float16” loads in torch.float16 etc. (default: auto)")
parser.add_argument("--attn_implementation", type=str, default="sdpa", help='The attention implementation to use in the model (if relevant). Can be any of "eager" (manual implementation of the attention), "sdpa" (using F.scaled_dot_product_attention), or "flash_attention_2" (using Dao-AILab/flash-attention). By default, if available, SDPA will be used for torch>=2.1.1. (default: sdpa)')
parser.add_argument("--device_map", type=str, default="auto", help="A string that is a valid device. E.g. “cuda” loads the model on the GPU, “cpu” loads it on the CPU. (default: auto)")

args = parser.parse_args()

quant_config = BitsAndBytesConfig(
    load_in_8bit=args.quant_load_in_8bit,
    load_in_4bit=args.quant_load_in_4bit,
    llm_int8_threshold=args.quant_llm_int8_threshold,
    llm_int8_skip_modules=args.quant_llm_int8_skip_modules,
    llm_int8_enable_fp32_cpu_offload=args.quant_llm_int8_enable_fp32_cpu_offload,
    llm_int8_has_fp16_weight=args.quant_llm_int8_has_fp16_weight,
    bnb_4bit_compute_dtype=args.quant_bnb_4bit_compute_dtype,
    bnb_4bit_quant_type=args.quant_bnb_4bit_quant_type,
    bnb_4bit_use_double_quant=args.quant_bnb_4bit_use_double_quant,
    bnb_4bit_quant_storage=args.quant_bnb_4bit_quant_storage
)

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(args.model)

if args.quant:
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        quantization_config=quant_config,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class StoppingCriteriaSub(StoppingCriteria):
    # https://discuss.huggingface.co/t/implimentation-of-stopping-criteria-list/20040/13
    def __init__(self, stop_words, tokenizer):
        super().__init__()
        self.stop_words = stop_words
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_word in self.stop_words:
            if self.tokenizer.decode(input_ids[0][-len(self.tokenizer.encode(stop_word)):]).strip() == stop_word:
                return True
        return False

class TokenizeRequest(BaseModel):
    input: str
    add_special_tokens: Optional[bool] = False

class DetokenizeRequest(BaseModel):
    tokens: List[int]
    skip_special_tokens: Optional[bool] = True

class ChatCompletionRequest(BaseModel):
    messages: List[dict]
    temperature: float = 1.
    top_p: float = 1.
    stop: Optional[List[str]] = None
    seed: Optional[int] = None
    max_tokens: Optional[int] = 2048
    max_tokens_forcing: Optional[int] = None
    top_k: int = 50
    logprobs: bool = False
    do_sample: bool = True
    num_beams: int = 1
    num_beam_groups: int = 1
    eos_token_id: Optional[int] = None

@app.post("/chat")
def chat_completions(request: ChatCompletionRequest):
    chat = request.messages
    #@TODO: is there a way to assert that the loaded model has chat capabilities?
    inputs = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inputs, return_tensors="pt").to(model.device)

    generation_config = {
        "max_length": request.max_tokens,
        "max_new_tokens": request.max_tokens_forcing,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "do_sample": request.do_sample,
        "num_beams": request.num_beams,
        "num_beam_groups": request.num_beam_groups,
        "eos_token_id": request.eos_token_id if request.eos_token_id is not None else tokenizer.eos_token_id,
        "output_logits": request.logprobs,
        "return_dict_in_generate": True,
    }

    if request.stop:
        generation_config["stopping_criteria"] = StoppingCriteriaList([StoppingCriteriaSub(stop_words=request.stop, tokenizer=tokenizer)])

    if request.seed:
        set_seed(request.seed)

    outputs = model.generate(**inputs, **generation_config)

    new_tokens = outputs.sequences[0][inputs.input_ids.shape[-1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": generated_text
                }
            },
        ],
        "metadata": {
            "model": args.model,
            "model_name": model.config.name_or_path,
            "model_class": model.__class__.__name__,
            "model_device": model.device.type,
            "model_vocab_size": model.config.vocab_size,
            "model_output_scores": request.logprobs,
            "model_do_sample": request.do_sample,
            "model_temperature": request.temperature,
            "model_top_p": request.top_p,
            "model_top_k": request.top_k,
            "model_max_tokens": request.max_tokens,
            "model_seed": request.seed,
            "model_stopping_criteria": request.stop,
            "model_eos_token_id": request.eos_token_id,
            "logits": [logits.tolist() for logits in outputs.logits],
            "model_input": chat,
            "model_chat_format": tokenizer.chat_template,
        }
    }

    return response

@app.post("/tokenize")
def tokenize(request: TokenizeRequest):
    tokens = tokenizer.encode(request.input, add_special_tokens=request.add_special_tokens)
    return {"tokens": tokens}

@app.post("/detokenize")
def detokenize(request: DetokenizeRequest):
    text = tokenizer.decode(request.tokens, skip_special_tokens=request.skip_special_tokens)
    return {"text": text}

def huggingface_server():
    import uvicorn
    from functools import partial
    command = partial(uvicorn.run, app)
    return command, args
