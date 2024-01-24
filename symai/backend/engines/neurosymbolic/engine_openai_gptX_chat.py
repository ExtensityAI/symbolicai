import logging
import re
import tiktoken
import openai

from typing import List, Optional

from ...base import Engine
from ...mixin.openai import OpenAIMixin
from ...settings import SYMAI_CONFIG
from ....utils import encode_frames_file
from ....misc.console import ConsoleStyle
from ....symbol import Symbol


logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


class InvalidRequestErrorRemedyChatStrategy:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, engine, error, callback, argument):
        openai_kwargs = {}
        kwargs              = argument.kwargs
        prompts_            = argument.prop.prepared_input

        # send prompt to GPT-X Chat-based
        stop                = kwargs['stop'] if 'stop' in kwargs else None
        model               = kwargs['model'] if 'model' in kwargs else None

        msg = str(error)
        handle = None
        try:
            if "This model's maximum context length is" in msg:
                handle = 'type1'
                max_ = engine.max_tokens
                usr = msg.split('tokens. ')[1].split(' ')[-1]
                overflow_tokens = int(usr) - int(max_)
            elif "is less than the minimum" in msg:
                handle = 'type2'
                # extract number until 'is'
                msg_ = msg.split("is less than the minimum")[0]
                # remove until the first `-`
                msg_ = msg_.split(': "-')[-1]
                overflow_tokens = int(msg_)
            else:
                raise Exception(msg) from error
        except Exception as e:
            raise e from error

        prompts = [p for p in prompts_ if p['role'] == 'user']
        system_prompt = [p for p in prompts_ if p['role'] == 'system']
        if handle == 'type1':
            truncated_content_ = [p['content'][overflow_tokens:] for p in prompts]
            truncated_prompts_ = [{'role': p['role'], 'content': c} for p, c in zip(prompts, truncated_content_)]
            with ConsoleStyle('warn') as console:
                console.print(f"WARNING: Overflow tokens detected. Reducing prompt size by {overflow_tokens} characters.")
        elif handle == 'type2':
            user_prompts = [p['content'] for p in prompts]
            new_prompt   = [*system_prompt]
            new_prompt.extend([{'role': p['role'], 'content': c} for p, c in zip(prompts, user_prompts)])
            overflow_tokens = engine.compute_required_tokens(new_prompt) - int(engine.max_tokens * 0.70)
            if overflow_tokens > 0:
                print('WARNING: Overflow tokens detected. Reducing prompt size to 70% of max_tokens.')
                for i, content in enumerate(user_prompts):
                    token_ids = engine.tokenizer.encode(content)
                    if overflow_tokens >= len(token_ids):
                        overflow_tokens -= len(token_ids)
                        user_prompts[i] = ''
                    else:
                        new_content = engine.tokenizer.decode(token_ids[:-overflow_tokens])
                        user_prompts[i] = new_content
                        overflow_tokens = 0
                        break

            new_prompt = [*system_prompt]
            new_prompt.extend([{'role': p['role'], 'content': c} for p, c in zip(prompts, user_prompts)])
            assert engine.compute_required_tokens(new_prompt) <= engine.max_tokens, \
                f"Token overflow: prompts exceed {engine.max_tokens} tokens after truncation"

            truncated_prompts_ = [{'role': p['role'], 'content': c.strip()} for p, c in zip(prompts, user_prompts) if c.strip()]
        else:
            raise Exception('Invalid handle case for remedy strategy.') from error

        truncated_prompts_ = [*system_prompt, *truncated_prompts_]

        # convert map to list of strings
        max_tokens          = kwargs['max_tokens'] if 'max_tokens' in kwargs else engine.compute_remaining_tokens(truncated_prompts_)
        temperature         = kwargs['temperature'] if 'temperature' in kwargs else 1
        frequency_penalty   = kwargs['frequency_penalty'] if 'frequency_penalty' in kwargs else 0
        presence_penalty    = kwargs['presence_penalty'] if 'presence_penalty' in kwargs else 0
        top_p               = kwargs['top_p'] if 'top_p' in kwargs else 1
        functions           = kwargs['functions'] if 'functions' in kwargs else None
        function_call       = "auto" if functions is not None else None

        if stop is not None:
            openai_kwargs['stop'] = stop
        if functions is not None:
            openai_kwargs['functions'] = functions
        if function_call is not None:
            openai_kwargs['function_call'] = function_call

        return callback(model=model,
                        messages=truncated_prompts_,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        top_p=top_p,
                        n=1,
                        **openai_kwargs)


class GPTXChatEngine(Engine, OpenAIMixin):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__()
        logger              = logging.getLogger('openai')
        logger.setLevel(logging.WARNING)
        self.config         = SYMAI_CONFIG
        openai.api_key      = self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] if api_key is None else api_key
        self.model          = self.config['NEUROSYMBOLIC_ENGINE_MODEL'] if model is None else model
        self.tokenizer      = tiktoken.encoding_for_model(self.model)
        self.pricing        = self.api_pricing()
        self.max_tokens     = self.api_max_tokens() - 100 # TODO: account for tolerance. figure out how their magic number works to compute reliably the precise max token size
        self.seed           = None
        self.except_remedy  = None

    def id(self) -> str:
        if   self.config['NEUROSYMBOLIC_ENGINE_MODEL'] and \
            (self.config['NEUROSYMBOLIC_ENGINE_MODEL'].startswith('gpt-3.5') or \
             self.config['NEUROSYMBOLIC_ENGINE_MODEL'].startswith('gpt-4')):
            return 'neurosymbolic'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'NEUROSYMBOLIC_ENGINE_API_KEY' in kwargs:
            openai.api_key = kwargs['NEUROSYMBOLIC_ENGINE_API_KEY']
        if 'NEUROSYMBOLIC_ENGINE_MODEL' in kwargs:
            self.model     = kwargs['NEUROSYMBOLIC_ENGINE_MODEL']
        if 'seed' in kwargs:
            self.seed      = kwargs['seed']
        if 'except_remedy' in kwargs:
            self.except_remedy = kwargs['except_remedy']

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

    def forward(self, argument):
        kwargs              = argument.kwargs
        prompts_            = argument.prop.prepared_input

        openai_kwargs = {}

        # send prompt to GPT-X Chat-based
        stop                = kwargs['stop'] if 'stop' in kwargs else None
        model               = kwargs['model'] if 'model' in kwargs else self.model
        seed                = kwargs['seed'] if 'seed' in kwargs else self.seed

        # convert map to list of strings
        max_tokens          = kwargs['max_tokens'] if 'max_tokens' in kwargs else self.compute_remaining_tokens(prompts_)
        temperature         = kwargs['temperature'] if 'temperature' in kwargs else 1
        frequency_penalty   = kwargs['frequency_penalty'] if 'frequency_penalty' in kwargs else 0
        presence_penalty    = kwargs['presence_penalty'] if 'presence_penalty' in kwargs else 0
        top_p               = kwargs['top_p'] if 'top_p' in kwargs else 1
        except_remedy       = kwargs['except_remedy'] if 'except_remedy' in kwargs else self.except_remedy
        functions           = kwargs['functions'] if 'functions' in kwargs else None
        function_call       = "auto" if functions is not None else None

        if stop is not None:
            openai_kwargs['stop'] = stop
        if functions is not None:
            openai_kwargs['functions'] = functions
        if function_call is not None:
            openai_kwargs['function_call'] = function_call
        if seed is not None:
            openai_kwargs['seed'] = seed

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

        except Exception as e:
            if openai.api_key is None or openai.api_key == '':
                msg = 'OpenAI API key is not set. Please set it in the config file or pass it as an argument to the command method.'
                logging.error(msg)
                raise Exception(msg) from e

            callback = openai.chat.completions.create
            kwargs['model'] = kwargs['model'] if 'model' in kwargs else self.model
            if except_remedy is not None:
                res = except_remedy(self, e, callback, argument)
            else:
                try:
                    # implicit remedy strategy
                    except_remedy = InvalidRequestErrorRemedyChatStrategy()
                    res = except_remedy(self, e, callback, argument)
                except Exception as e2:
                    ex = Exception(f'Failed to handle exception: {e}. Also failed implicit remedy strategy after retry: {e2}')
                    raise ex from e

        metadata = {}

        rsp    = [r.message.content for r in res.choices]
        output = rsp if isinstance(prompts_, list) else rsp[0]
        return output, metadata

    def prepare(self, argument):
        if argument.prop.raw_input:
            if not argument.prop.processed_input:
                raise ValueError('Need to provide a prompt instruction to the engine if raw_input is enabled.')
            value = argument.prop.processed_input
            # convert to dict if not already
            if type(value) != list:
                if type(value) != dict:
                    value = {'role': 'user', 'content': str(value)}
                value = [value]
            argument.prop.prepared_input = value
            return

        _non_verbose_output = """[META INSTRUCTIONS START]\nYou do not output anything else, like verbose preambles or post explanation, such as "Sure, let me...", "Hope that was helpful...", "Yes, I can help you with that...", etc. Consider well formatted output, e.g. for sentences use punctuation, spaces etc. or for code use indentation, etc. Never add meta instructions information to your output!\n"""
        user:   str = ""
        system: str = ""

        if argument.prop.suppress_verbose_output:
            system += _non_verbose_output
        system = f'{system}\n' if system and len(system) > 0 else ''

        ref = argument.prop.instance
        static_ctxt, dyn_ctxt = ref.global_context
        if len(static_ctxt) > 0:
            system += f"[STATIC CONTEXT]\n{static_ctxt}\n\n"

        if len(dyn_ctxt) > 0:
            system += f"[DYNAMIC CONTEXT]\n{dyn_ctxt}\n\n"

        payload = argument.prop.payload
        if argument.prop.payload:
            system += f"[ADDITIONAL CONTEXT]\n{str(payload)}\n\n"

        examples: List[str] = argument.prop.examples
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
        if self.model == 'gpt-4-vision-preview' and '<<vision:' in str(argument.prop.processed_input):
            parts = extract_pattern(str(argument.prop.processed_input))
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

        if argument.prop.prompt is not None and len(argument.prop.prompt) > 0:
            val = str(argument.prop.prompt)
            if len(image_files) > 0:
                val = remove_pattern(val)
            system += f"[INSTRUCTION]\n{val}"

        suffix: str = str(argument.prop.processed_input)
        if len(image_files) > 0:
            suffix = remove_pattern(suffix)

        if '[SYSTEM_INSTRUCTION::]: <<<' in suffix and argument.prop.parse_system_instructions:
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

        if argument.prop.template_suffix:
            user += f"\n[[PLACEHOLDER]]\n{str(argument.prop.template_suffix)}\n\n"
            user += f"Only generate content for the placeholder `[[PLACEHOLDER]]` following the instructions and context information. Do NOT write `[[PLACEHOLDER]]` or anything else in your output.\n\n"

        images = [{ 'type': 'image', "image_url": { "url": file, "detail": "auto" }} for file in image_files]
        if self.model == 'gpt-4-vision-preview':
            user_prompt = { "role": "user", "content": [
                *images,
                { 'type': 'text', 'text': user }
            ]}
        else:
            user_prompt = { "role": "user", "content": user }

        argument.prop.prepared_input = [
            { "role": "system", "content": system },
            user_prompt,
        ]
