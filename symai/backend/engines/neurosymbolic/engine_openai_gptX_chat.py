import logging
import re
import tiktoken
import openai

from typing import List, Optional

from ...base import Engine
from ...mixin.openai import OpenAIMixin
from ...settings import SYMAI_CONFIG
from ....utils import encode_frames_file, CustomUserWarning
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
        kwargs              = argument.kwargs
        prompts_            = argument.prop.prepared_input

        msg = str(error)
        handle = None
        try:
            if "This model's maximum context length is" in msg:
                handle = 'type1'
                max_ = engine.max_context_tokens
                usr = msg.split('tokens. ')[1].split(' ')[-1]
                overflow_tokens = int(usr) - int(max_)
            elif "is less than the minimum" in msg:
                handle = 'type2'
                overflow_tokens = engine.max_response_tokens
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
            overflow_tokens = engine.compute_required_tokens(new_prompt) - int(engine.max_context_tokens * 0.70)
            if overflow_tokens > 0:
                CustomUserWarning(f'WARNING: Overflow tokens detected. Reducing prompt size to 70% of model context size ({engine.max_context_tokens}).')
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
            assert engine.compute_required_tokens(new_prompt) <= engine.max_context_tokens, \
                f"Token overflow: prompts exceed {engine.max_context_tokens} tokens after truncation"

            truncated_prompts_ = [{'role': p['role'], 'content': c.strip()} for p, c in zip(prompts, user_prompts) if c.strip()]
        else:
            raise Exception('Invalid handle case for remedy strategy.') from error

        truncated_prompts_ = [*system_prompt, *truncated_prompts_]

        model             = kwargs.get('model',             engine.model)
        seed              = kwargs.get('seed',              engine.seed)
        max_tokens        = kwargs.get('max_tokens',        engine.compute_remaining_tokens(truncated_prompts_))
        stop              = kwargs.get('stop')
        temperature       = kwargs.get('temperature',       1)
        frequency_penalty = kwargs.get('frequency_penalty', 0)
        presence_penalty  = kwargs.get('presence_penalty',  0)
        top_p             = kwargs.get('top_p',             1)
        n                 = kwargs.get('n',                 1)
        logit_bias        = kwargs.get('logit_bias')
        logprobs          = kwargs.get('logprobs',          False)
        top_logprobs      = kwargs.get('top_logprobs')
        tools             = kwargs.get('tools')
        tool_choice       = kwargs.get('tool_choice')
        response_format   = kwargs.get('response_format')

        return callback(
                model=model,
                messages=truncated_prompts_,
                max_tokens=max_tokens,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                top_p=top_p,
                n=n,
                logit_bias=logit_bias,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                seed=seed,
                stop=stop
            )


class GPTXChatEngine(Engine, OpenAIMixin):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__()
        logger = logging.getLogger('openai')
        logger.setLevel(logging.WARNING)
        self.config = SYMAI_CONFIG
        if self.id() != 'neurosymbolic':
            return # do not initialize if not neurosymbolic; avoids conflict with llama.cpp check in EngineRepository.register_from_package
        openai.api_key           = self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] if api_key is None else api_key
        self.model               = self.config['NEUROSYMBOLIC_ENGINE_MODEL'] if model is None else model
        self.tokenizer           = tiktoken.encoding_for_model(self.model)
        self.pricing             = self.api_pricing()
        self.max_context_tokens  = self.api_max_context_tokens()
        self.max_response_tokens = self.api_max_response_tokens()
        self.seed                = None
        self.except_remedy       = None

    def id(self) -> str:
        if   self.config.get('NEUROSYMBOLIC_ENGINE_MODEL') and \
            (self.config.get('NEUROSYMBOLIC_ENGINE_MODEL').startswith('gpt-3.5') or \
             self.config.get('NEUROSYMBOLIC_ENGINE_MODEL').startswith('gpt-4')):
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

    def compute_required_tokens(self, messages):
        """Return the number of tokens used by a list of messages."""

        if self.model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            "gpt-4-turbo",
            "gpt-4o"
            }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif self.model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1    # if there's a name, the role is omitted
        elif self.model == "gpt-3.5-turbo":
            CustomUserWarning("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            tokens_per_message = 3
            tokens_per_name = 1
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")
        elif self.model == "gpt-4":
            tokens_per_message = 3
            tokens_per_name = 1
            CustomUserWarning("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            self.tokenizer = tiktoken.encoding_for_model("gpt-4-0613")
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {self.model}. See https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken for information on how messages are converted to tokens."""
            )

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if type(value) == str:
                    num_tokens += len(self.tokenizer.encode(value, disallowed_special=()))
                else:
                    for v in value:
                        if v['type'] == 'text':
                            num_tokens += len(self.tokenizer.encode(v['text'], disallowed_special=()))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def compute_remaining_tokens(self, prompts: list) -> int:
        val = self.compute_required_tokens(prompts)
        #@NOTE: this will obviously fail if val is greater than max_context_tokens
        #       the remedy strategy should handle this case
        return min(self.max_context_tokens - val, self.max_response_tokens)

    def forward(self, argument):
        kwargs        = argument.kwargs
        prompts_      = argument.prop.prepared_input

        model             = kwargs.get('model',             self.model)
        seed              = kwargs.get('seed',              self.seed)
        except_remedy     = kwargs.get('except_remedy',     self.except_remedy)
        max_tokens        = kwargs.get('max_tokens',        self.compute_remaining_tokens(prompts_))
        stop              = kwargs.get('stop',              '')
        temperature       = kwargs.get('temperature',       1)
        frequency_penalty = kwargs.get('frequency_penalty', 0)
        presence_penalty  = kwargs.get('presence_penalty',  0)
        top_p             = kwargs.get('top_p',             1)
        n                 = kwargs.get('n',                 1)
        logit_bias        = kwargs.get('logit_bias')
        logprobs          = kwargs.get('logprobs')
        top_logprobs      = kwargs.get('top_logprobs')
        tools             = kwargs.get('tools')
        tool_choice       = kwargs.get('tool_choice')
        response_format   = kwargs.get('response_format')

        try:
            res = openai.chat.completions.create(
                    model=model,
                    messages=prompts_,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    top_p=top_p,
                    n=n,
                    logit_bias=logit_bias,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    seed=seed,
                    stop=stop
                )

        except Exception as e:
            if openai.api_key is None or openai.api_key == '':
                msg = 'OpenAI API key is not set. Please set it in the config file or pass it as an argument to the command method.'
                logging.error(msg)
                if self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] is None or self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] == '':
                    raise Exception(msg) from e
                openai.api_key = self.config['NEUROSYMBOLIC_ENGINE_API_KEY']

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

        metadata = {'raw_output': res}

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

        _non_verbose_output = """<META_INSTRUCTION/>\nYou do not output anything else, like verbose preambles or post explanation, such as "Sure, let me...", "Hope that was helpful...", "Yes, I can help you with that...", etc. Consider well formatted output, e.g. for sentences use punctuation, spaces etc. or for code use indentation, etc. Never add meta instructions information to your output!\n\n"""
        user:   str = ""
        system: str = ""

        if argument.prop.suppress_verbose_output:
            system += _non_verbose_output
        system = f'{system}\n' if system and len(system) > 0 else ''

        if argument.prop.response_format:
            system += '<JSON_RESPONSE/>\n You will output JSON!\n\n'

        ref = argument.prop.instance
        static_ctxt, dyn_ctxt = ref.global_context
        if len(static_ctxt) > 0:
            system += f"<STATIC CONTEXT/>\n{static_ctxt}\n\n"

        if len(dyn_ctxt) > 0:
            system += f"<DYNAMIC CONTEXT/>\n{dyn_ctxt}\n\n"

        payload = argument.prop.payload
        if argument.prop.payload:
            system += f"<ADDITIONAL CONTEXT/>\n{str(payload)}\n\n"

        examples: List[str] = argument.prop.examples
        if examples and len(examples) > 0:
            system += f"<EXAMPLES/>\n{str(examples)}\n\n"

        def extract_pattern(text):
            pattern = r'<<vision:(.*?):>>'
            return re.findall(pattern, text)

        def remove_pattern(text):
            pattern = r'<<vision:(.*?):>>'
            return re.sub(pattern, '', text)

        image_files = []
        # pre-process prompt if contains image url
        if (self.model == 'gpt-4-vision-preview' or \
            self.model == 'gpt-4-turbo-2024-04-09' or \
            self.model == 'gpt-4-turbo' or \
            self.model == 'gpt-4o') \
            and '<<vision:' in str(argument.prop.processed_input):

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
            system += f"<INSTRUCTION/>\n{val}\n\n"

        suffix: str = str(argument.prop.processed_input)
        if len(image_files) > 0:
            suffix = remove_pattern(suffix)

        if '[SYSTEM_INSTRUCTION::]: <<<' in suffix and argument.prop.parse_system_instructions:
            parts = suffix.split('\n>>>\n')
            # first parts are the system instructions
            c = 0
            for i, p in enumerate(parts):
                if 'SYSTEM_INSTRUCTION' in p:
                    system += f"<{p}/>\n\n"
                    c += 1
                else:
                    break
            # last part is the user input
            suffix = '\n>>>\n'.join(parts[c:])
        user += f"{suffix}"

        if argument.prop.template_suffix:
            system += f' You will only generate content for the placeholder `{str(argument.prop.template_suffix)}` following the instructions and the provided context information.\n\n'

        if self.model == 'gpt-4-vision-preview':
           images = [{ 'type': 'image', "image_url": { "url": file }} for file in image_files]
           user_prompt = { "role": "user", "content": [
                *images,
                { 'type': 'text', 'text': user }
            ]}
        elif self.model == 'gpt-4-turbo-2024-04-09' or \
             self.model == 'gpt-4-turbo' or \
             self.model == 'gpt-4o':

            images = [{ 'type': 'image_url', "image_url": { "url": file }} for file in image_files]
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
