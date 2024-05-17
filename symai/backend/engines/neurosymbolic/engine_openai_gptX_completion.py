import logging
import openai
import tiktoken

from typing import List, Optional

from ...base import Engine
from ...mixin.openai import OpenAIMixin
from ...settings import SYMAI_CONFIG
from ....utils import CustomUserWarning
from ....misc.console import ConsoleStyle


logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


class InvalidRequestErrorRemedyCompletionStrategy:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, engine, error, callback, argument):
        openai_kwargs = {}
        kwargs   = argument.kwargs
        prompts_ = argument.prop.prepared_input
        # send prompt to GPT-X Completion-based
        stop  = kwargs['stop'] if 'stop' in kwargs else None
        model = kwargs['model'] if 'model' in kwargs else None

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

        # unify the format to use same remedy strategy for both chat and completion
        values = prompts_[0].replace('---------SYSTEM BEHAVIOR--------\n', '').split('\n\n---------USER REQUEST--------\n')
        prompts_ = [{'role': 'system', 'content': values[0]}, {'role': 'user', 'content': values[1]}]

        prompts = [p for p in prompts_ if p['role'] == 'user']
        system_prompt = [p for p in prompts_ if p['role'] == 'system']

        def compute_required_tokens(prompts: dict) -> int:
            # iterate over prompts and compute number of tokens
            prompts_ = [role['content'] for role in prompts]
            prompt = ''.join(prompts_)
            val = len(engine.tokenizer.encode(prompt, disallowed_special=()))
            return val

        def compute_remaining_tokens(prompts: list) -> int:
            val = compute_required_tokens(prompts)
            return int((engine.max_context_tokens - val) * 0.99)

        if handle == 'type1':
            truncated_content_ = [p['content'][overflow_tokens:] for p in prompts]
            truncated_prompts_ = [{'role': p['role'], 'content': c} for p, c in zip(prompts, truncated_content_)]
            with ConsoleStyle('warn') as console:
                console.print(f"WARNING: Overflow tokens detected. Reducing prompt size by {overflow_tokens} characters.")
        elif handle == 'type2':
            user_prompts = [p['content'] for p in prompts]
            new_prompt   = [*system_prompt]
            new_prompt.extend([{'role': p['role'], 'content': c} for p, c in zip(prompts, user_prompts)])
            overflow_tokens = compute_required_tokens(new_prompt) - int(engine.max_context_tokens * 0.70)
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
            assert compute_required_tokens(new_prompt) <= engine.max_context_tokens, \
                f"Token overflow: prompts exceed {engine.max_context_tokens} tokens after truncation"

            truncated_prompts_ = [{'role': p['role'], 'content': c.strip()} for p, c in zip(prompts, user_prompts) if c.strip()]
        else:
            raise Exception('Invalid handle case for remedy strategy.') from error

        truncated_prompts_ = [*system_prompt, *truncated_prompts_]

        # convert map to list of strings
        max_tokens          = kwargs['max_tokens'] if 'max_tokens' in kwargs else compute_remaining_tokens(truncated_prompts_)
        temperature         = kwargs['temperature'] if 'temperature' in kwargs else 1
        frequency_penalty   = kwargs['frequency_penalty'] if 'frequency_penalty' in kwargs else 0
        presence_penalty    = kwargs['presence_penalty'] if 'presence_penalty' in kwargs else 0
        top_p               = kwargs['top_p'] if 'top_p' in kwargs else 1
        suffix              = kwargs['template_suffix'] if 'template_suffix' in kwargs else None

        system = truncated_prompts_[0]['content']
        user = truncated_prompts_[1]['content']
        truncated_prompts_ = [f'---------SYSTEM BEHAVIOR--------\n{system}\n\n---------USER REQUEST--------\n{user}']

        if stop is not None:
            openai_kwargs['stop'] = stop

        return callback(model=model,
                        prompt=truncated_prompts_,
                        suffix=suffix,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        top_p=top_p,
                        n=1,
                        **openai_kwargs)



class GPTXCompletionEngine(Engine, OpenAIMixin):
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
        self.except_remedy       = None

    def id(self) -> str:
        if   self.config['NEUROSYMBOLIC_ENGINE_MODEL'] and \
            (self.config['NEUROSYMBOLIC_ENGINE_MODEL'].startswith('text-') or \
             self.config['NEUROSYMBOLIC_ENGINE_MODEL'].startswith('davinci') or \
             self.config['NEUROSYMBOLIC_ENGINE_MODEL'].startswith('curie') or \
             self.config['NEUROSYMBOLIC_ENGINE_MODEL'].startswith('babbage') or \
             self.config['NEUROSYMBOLIC_ENGINE_MODEL'].startswith('ada')):
            return 'neurosymbolic'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'NEUROSYMBOLIC_ENGINE_API_KEY' in kwargs:
            openai.api_key = kwargs['NEUROSYMBOLIC_ENGINE_API_KEY']
        if 'NEUROSYMBOLIC_ENGINE_MODEL' in kwargs:
            self.model     = kwargs['NEUROSYMBOLIC_ENGINE_MODEL']
        if 'except_remedy' in kwargs:
            self.except_remedy = kwargs['except_remedy']

    def compute_required_tokens(self, prompts: list) -> int:
       # iterate over prompts and compute number of tokens
        prompt = prompts[0] # index 0 is correct since we only have one prompt in legacy mode
        val = len(self.tokenizer.encode(prompt, disallowed_special=()))
        return val

    def compute_remaining_tokens(self, prompts: list) -> int:
        val = self.compute_required_tokens(prompts)
        return min(self.max_context_tokens - val, self.max_response_tokens)

    def forward(self, argument):
        kwargs              = argument.kwargs
        prompts_            = argument.prop.prepared_input

        # send prompt to GPT-3
        max_tokens          = kwargs['max_tokens'] if 'max_tokens' in kwargs else self.compute_remaining_tokens(prompts_)
        stop                = kwargs['stop'] if 'stop' in kwargs else None
        model               = kwargs['model'] if 'model' in kwargs else self.model
        temperature         = kwargs['temperature'] if 'temperature' in kwargs else 0.7
        frequency_penalty   = kwargs['frequency_penalty'] if 'frequency_penalty' in kwargs else 0
        presence_penalty    = kwargs['presence_penalty'] if 'presence_penalty' in kwargs else 0
        top_p               = kwargs['top_p'] if 'top_p' in kwargs else 1
        except_remedy       = kwargs['except_remedy'] if 'except_remedy' in kwargs else self.except_remedy

        try:
            res = openai.completions.create(model=model,
                                            prompt=prompts_,
                                            max_tokens=max_tokens,
                                            temperature=temperature,
                                            frequency_penalty=frequency_penalty,
                                            presence_penalty=presence_penalty,
                                            top_p=top_p,
                                            stop=stop,
                                            n=1)
        except Exception as e:
            callback = openai.completions.create
            kwargs['model'] = kwargs['model'] if 'model' in kwargs else self.model
            if except_remedy is not None:
                res = except_remedy(self, e, callback, argument)
            else:
                try:
                    # implicit remedy strategy
                    except_remedy = InvalidRequestErrorRemedyCompletionStrategy()
                    res = except_remedy(self, e, callback, argument)
                except Exception as e2:
                    ex = Exception(f'Failed to handle exception: {e}. Also failed implicit remedy strategy after retry: {e2}')
                    raise ex from e

        metadata = {}
        # TODO: remove system behavior and user request from output. consider post-processing
        def replace_verbose(rsp):
            rsp = rsp.replace('---------SYSTEM BEHAVIOR--------\n', '')
            rsp = rsp.replace('\n\n---------USER REQUEST--------\n', '')
            return rsp

        rsp    = [replace_verbose(r.text) for r in res.choices]
        output = rsp if isinstance(prompts_, list) else rsp[0]
        return output, metadata

    def prepare(self, argument):
        if argument.prop.raw_input:
            if not argument.prop.processed_input:
                raise ValueError('Need to provide a prompt instruction to the engine if raw_input is enabled.')
            value = argument.prop.processed_input
            if type(value) is not list:
                value = [str(value)]
            argument.prop.prepared_input = value
            return

        _non_verbose_output = """[META INSTRUCTIONS START]\nYou do not output anything else, like verbose preambles or post explanation, such as "Sure, let me...", "Hope that was helpful...", "Yes, I can help you with that...", etc. Consider well formatted output, e.g. for sentences use punctuation, spaces etc. or for code use indentation, etc. Never add meta instructions information to your output!\n"""

        user:   str = ""
        system: str = ""

        if argument.prop.suppress_verbose_output:
            system  += _non_verbose_output
        system = f'{system}\n' if system and len(system) > 0 else ''

        ref = argument.prop.instance
        static_ctxt, dyn_ctxt = ref.global_context
        if len(static_ctxt) > 0:
            system += f"[STATIC CONTEXT]\n{static_ctxt}\n\n"

        if len(dyn_ctxt) > 0:
            system += f"[DYNAMIC CONTEXT]\n{dyn_ctxt}\n\n"

        payload = argument.prop.payload
        if payload is not None:
            system += f"[ADDITIONAL CONTEXT]\n{payload}\n\n"

        examples: List[str] = argument.prop.examples
        if examples and len(examples) > 0:
            system += f"[EXAMPLES]\n{str(examples)}\n\n"

        if argument.prop.prompt is not None and len(argument.prop.prompt) > 0:
            val = str(argument.prop.prompt)
            system += f"[INSTRUCTION]\n{val}"

        suffix: str = str(argument.prop.processed_input)
        if '=>' in suffix:
            user += f"[LAST TASK]\n"

        if '[SYSTEM_INSTRUCTION::]: <<<' in suffix and argument.prop.parse_system_instructions:
            parts = suffix.split('\n>>>\n')
            # first parts are the system instructions
            for p in parts[:-1]:
                system += f"{p}\n"
            # last part is the user input
            suffix = parts[-1]
        user += f"{suffix}"

        if argument.prop.template_suffix is not None:
            user += f"\n[[PLACEHOLDER]]\n{argument.prop.template_suffix}\n\n"
            user += f"Only generate content for the placeholder `[[PLACEHOLDER]]` following the instructions and context information. Do NOT write `[[PLACEHOLDER]]` or anything else in your output.\n\n"

        argument.prop.prepared_input = [f'---------SYSTEM BEHAVIOR--------\n{system}\n\n---------USER REQUEST--------\n{user}']
