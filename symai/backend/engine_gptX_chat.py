import logging
from typing import List
import re
import sys
import tiktoken
import openai

from .base import Engine
from .mixin.openai import OpenAIMixin
from .settings import SYMAI_CONFIG
from ..strategy import InvalidRequestErrorRemedyChatStrategy
from ..utils import encode_frames_file


logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


class GPTXChatEngine(Engine, OpenAIMixin):
    def __init__(self):
        super().__init__()
        logger = logging.getLogger('openai')
        logger.setLevel(logging.WARNING)
        config          = SYMAI_CONFIG
        openai.api_key  = config['NEUROSYMBOLIC_ENGINE_API_KEY']
        self.model      = config['NEUROSYMBOLIC_ENGINE_MODEL']
        self.tokenizer  = tiktoken.encoding_for_model(self.model)
        self.pricing    = self.api_pricing()
        self.max_tokens = self.api_max_tokens() - 100 # TODO: account for tolerance. figure out how their magic number works to compute reliably the precise max token size

    def command(self, wrp_params):
        super().command(wrp_params)
        if 'NEUROSYMBOLIC_ENGINE_API_KEY' in wrp_params:
            openai.api_key = wrp_params['NEUROSYMBOLIC_ENGINE_API_KEY']
        if 'NEUROSYMBOLIC_ENGINE_MODEL' in wrp_params:
            self.model = wrp_params['NEUROSYMBOLIC_ENGINE_MODEL']

    def compute_required_tokens(self, prompts: dict) -> int:
        # iterate over prompts and compute number of tokens
        prompts_ = [role['content'] for role in prompts]
        if self.model == 'gpt-4-vision-preview':
            eval_prompt = ''
            for p in prompts_:
                if type(p) == str:
                    eval_prompt += p
                else:
                    for p_ in p:
                        if p_['type'] == 'text':
                            eval_prompt += p_['text']
            prompt = eval_prompt
        else:
            prompt = ''.join(prompts_)
        val = len(self.tokenizer.encode(prompt, disallowed_special=()))
        return val

    def compute_remaining_tokens(self, prompts: list) -> int:
        val = self.compute_required_tokens(prompts)
        if 'gpt-4-1106-preview' == self.model or 'gpt-4-vision-preview' == self.model: # models can only output 4_096 tokens
            return min(int((self.max_tokens - val) * 0.99), 4_096)
        return int((self.max_tokens - val) * 0.99) # TODO: figure out how their magic number works to compute reliably the precise max token size

    def forward(self, prompts: List[str], *args, **kwargs) -> List[str]:
        prompts_            = prompts
        input_handler       = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((prompts_,))

        openai_kwargs = {}

        # send prompt to GPT-X Chat-based
        stop                = kwargs['stop'] if 'stop' in kwargs else None
        model               = kwargs['model'] if 'model' in kwargs else self.model

        # convert map to list of strings
        max_tokens          = kwargs['max_tokens'] if 'max_tokens' in kwargs else self.compute_remaining_tokens(prompts_)
        temperature         = kwargs['temperature'] if 'temperature' in kwargs else 1
        frequency_penalty   = kwargs['frequency_penalty'] if 'frequency_penalty' in kwargs else 0
        presence_penalty    = kwargs['presence_penalty'] if 'presence_penalty' in kwargs else 0
        top_p               = kwargs['top_p'] if 'top_p' in kwargs else 1
        except_remedy       = kwargs['except_remedy'] if 'except_remedy' in kwargs else None
        functions           = kwargs['functions'] if 'functions' in kwargs else None
        function_call       = "auto" if functions is not None else None

        if stop is not None:
            openai_kwargs['stop'] = stop
        if functions is not None:
            openai_kwargs['functions'] = functions
        if function_call is not None:
            openai_kwargs['function_call'] = function_call

        try:
            res = openai.chat.completions.create(model=model,
                                                 messages=prompts_,
                                                 max_tokens=max_tokens,
                                                 temperature=temperature,
                                                 frequency_penalty=frequency_penalty,
                                                 presence_penalty=presence_penalty,
                                                 top_p=top_p,
                                                 n=1,
                                                 **openai_kwargs)

            output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
            if output_handler:
                output_handler(res)
        except Exception as e:
            if openai.api_key is None or openai.api_key == '':
                msg = 'OpenAI API key is not set. Please set it in the config file or pass it as an argument to the command method.'
                logging.error(msg)
                raise Exception(msg) from e

            callback = openai.chat.completions.create
            kwargs['model'] = kwargs['model'] if 'model' in kwargs else self.model
            if except_remedy is not None:
                res = except_remedy(e, prompts_, callback, self, *args, **kwargs)
            else:
                try:
                    # implicit remedy strategy
                    except_remedy = InvalidRequestErrorRemedyChatStrategy()
                    res = except_remedy(e, prompts_, callback, self, *args, **kwargs)
                except Exception as e2:
                    ex = Exception(f'Failed to handle exception: {e}. Also failed implicit remedy strategy after retry: {e2}')
                    raise ex from e

        metadata = {}
        if 'metadata' in kwargs and kwargs['metadata']:
            metadata['kwargs'] = kwargs
            metadata['input']  = prompts_
            metadata['output'] = res
            metadata['model']  = model
            metadata['max_tokens'] = max_tokens
            metadata['temperature'] = temperature
            metadata['frequency_penalty'] = frequency_penalty
            metadata['presence_penalty'] = presence_penalty
            metadata['top_p'] = top_p
            metadata['except_remedy'] = except_remedy
            metadata['functions'] = functions
            metadata['function_call'] = function_call
            metadata['stop'] = stop

        rsp    = [r.message.content for r in res.choices]
        output = rsp if isinstance(prompts, list) else rsp[0]
        return output, metadata

    def prepare(self, args, kwargs, wrp_params):
        if 'raw_input' in wrp_params:
            wrp_params['prompts'] = wrp_params['raw_input']
            return

        disable_verbose_output = True if 'enable_verbose_output' not in wrp_params else not wrp_params['enable_verbose_output']
        _non_verbose_output = """[META INSTRUCTIONS START]\nYou do not output anything else, like verbose preambles or post explanation, such as "Sure, let me...", "Hope that was helpful...", "Yes, I can help you with that...", etc. Consider well formatted output, e.g. for sentences use punctuation, spaces etc. or for code use indentation, etc. Never add meta instructions information to your output!\n"""
        user:   str = ""
        system: str = ""

        if disable_verbose_output:
            system += _non_verbose_output
        system = f'{system}\n' if system and len(system) > 0 else ''

        ref = wrp_params['wrp_self']
        static_ctxt, dyn_ctxt = ref.global_context
        if len(static_ctxt) > 0:
            system += f"[STATIC CONTEXT]\n{static_ctxt}\n\n"

        if len(dyn_ctxt) > 0:
            system += f"[DYNAMIC CONTEXT]\n{dyn_ctxt}\n\n"

        payload = str(wrp_params['payload']) if 'payload' in wrp_params else None
        if payload is not None:
            system += f"[ADDITIONAL CONTEXT]\n{payload}\n\n"

        examples: List[str] = wrp_params['examples']
        if examples and len(examples) > 0:
            system += f"[EXAMPLES]\n{str(examples)}\n\n"

        def extract_pattern(text):
            pattern = r'<<vision:(.*?):>>'
            return re.findall(pattern, text)

        def remove_pattern(text):
            pattern = r'<<vision:(.*?):>>'
            return re.sub(pattern, '', text)

        image_files = []
        # pre-process prompt if contains image url
        if self.model == 'gpt-4-vision-preview' and '<<vision:' in str(wrp_params['processed_input']):
            parts = extract_pattern(str(wrp_params['processed_input']))
            for p in parts:
                img_ = p.strip()
                if img_.startswith('http'):
                    image_files.append(img_)
                elif img_.startswith('data:image'):
                    image_files.append(img_)
                else:
                    max_frames_spacing = 50
                    max_used_frames = 10
                    if img_.startswith('frames:'):
                        img_ = img_.replace('frames:', '')
                        max_used_frames, img_ = img_.split(':')
                        max_used_frames = int(max_used_frames)
                        if max_used_frames < 1 or max_used_frames > max_frames_spacing:
                            raise ValueError(f"Invalid max_used_frames value: {max_used_frames}. Expected value between 1 and {max_frames_spacing}")
                    buffer, ext = encode_frames_file(img_)
                    if len(buffer) > 1:
                        step = len(buffer) // max_frames_spacing # max frames spacing
                        frames = []
                        indices = list(range(0, len(buffer), step))[:max_used_frames]
                        for i in indices:
                            frames.append(f"data:image/{ext};base64,{buffer[i]}")
                        image_files.extend(frames)
                    elif len(buffer) == 1:
                        image_files.append(f"data:image/{ext};base64,{buffer[0]}")
                    else:
                        print('No frames found or error in encoding frames')

        if wrp_params['prompt'] is not None and len(wrp_params['prompt']) > 0 and ']: <<<' not in str(wrp_params['prompt']):
            val = str(wrp_params['prompt'])
            if len(image_files) > 0:
                val = remove_pattern(val)
            system += f"[INSTRUCTION]\n{val}"

        suffix: str = str(wrp_params['processed_input'])
        if len(image_files) > 0:
            suffix = remove_pattern(suffix)

        parse_system_instructions = False if 'parse_system_instructions' not in wrp_params else wrp_params['parse_system_instructions']
        if '[SYSTEM_INSTRUCTION::]: <<<' in suffix and parse_system_instructions:
            parts = suffix.split('\n>>>\n')
            # first parts are the system instructions
            c = 0
            for i, p in enumerate(parts):
                if 'SYSTEM_INSTRUCTION' in p:
                    system += f"{p}\n"
                    c += 1
                else:
                    break
            # last part is the user input
            suffix = '\n>>>\n'.join(parts[c:])
        user += f"{suffix}"

        template_suffix = str(wrp_params['template_suffix']) if 'template_suffix' in wrp_params else None
        if template_suffix:
            user += f"\n[[PLACEHOLDER]]\n{template_suffix}\n\n"
            user += f"Only generate content for the placeholder `[[PLACEHOLDER]]` following the instructions and context information. Do NOT write `[[PLACEHOLDER]]` or anything else in your output.\n\n"

        images = [{ 'type': 'image', "image_url": { "url": file, "detail": "auto" }} for file in image_files]
        if self.model == 'gpt-4-vision-preview':
            user_prompt = { "role": "user", "content": [
                *images,
                { 'type': 'text', 'text': user }
            ]}
        else:
            user_prompt = { "role": "user", "content": user }

        wrp_params['prompts'] = [
            { "role": "system", "content": system },
            user_prompt,
        ]

