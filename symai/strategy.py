import logging
from pydoc import locate

from .misc.console import ConsoleStyle
from .symbol import Expression


class Strategy(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def __new__(self, module: str, *args, **kwargs):
        module = module.lower()
        module = module.replace('-', '_')
        self._module = module
        self.module_path = f'symai.extended.strategies.{module}'
        return Strategy.load_module_class(self.module_path, self._module)(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def load_module_class(module_path, class_name):
        module_ = locate(module_path)
        return getattr(module_, class_name)


class InvalidRequestErrorRemedyChatStrategy:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, error, prompts_, callback, instance, *args, **kwargs):
        openai_kwargs = {}

        # send prompt to GPT-X Chat-based
        stop                = kwargs['stop'] if 'stop' in kwargs else None
        model               = kwargs['model'] if 'model' in kwargs else None

        msg = str(error)
        handle = None
        try:
            if "This model's maximum context length is" in msg:
                handle = 'type1'
                max_ = instance.max_tokens
                usr = msg.split('tokens. ')[1].split(' ')[-1]
                overflow_tokens = int(usr) - int(max_)
            elif "is less than the minimum" in msg:
                handle = 'type2'
                # extract number until 'is'
                msg_ = msg.split(' ')[0]
                overflow_tokens = int(msg_) * (-1)
            else:
                raise Exception(msg)
        except Exception as e:
            raise e

        prompts = [p for p in prompts_ if p['role'] == 'user']
        system_prompt = [p for p in prompts_ if p['role'] == 'system']
        if handle == 'type1':
            truncated_content_ = [p['content'][overflow_tokens:] for p in prompts]
            removed_content_ = [p['content'][:overflow_tokens] for p in prompts]
            truncated_prompts_ = [{'role': p['role'], 'content': c} for p, c in zip(prompts, truncated_content_)]
            with ConsoleStyle('warn') as console:
                console.print(f"WARNING: Overflow tokens detected. Reducing prompt size by {overflow_tokens} characters.")
        elif handle == 'type2':
            user_prompts = [p['content'] for p in prompts]
            # truncate until tokens are less than max_tokens * 0.70
            new_prompt = [*system_prompt]
            new_prompt.extend([{'role': p['role'], 'content': c} for p, c in zip(prompts, user_prompts)])
            while instance.compute_required_tokens(new_prompt) > instance.max_tokens * 0.70: # magic number
                user_prompts = [c[overflow_tokens:] for c in user_prompts]
                new_prompt = [*system_prompt]
                new_prompt.extend([{'role': p['role'], 'content': c} for p, c in zip(prompts, user_prompts)])
            with ConsoleStyle('warn') as console:
                console.print(f"WARNING: Overflow tokens detected. Reducing prompt size to {70}% of max_tokens.")
            truncated_prompts_ = [{'role': p['role'], 'content': c} for p, c in zip(prompts, user_prompts)]
        else:
            raise Exception('Invalid handle case for remedy strategy.')

        truncated_prompts_ = [*system_prompt, *truncated_prompts_]

        # convert map to list of strings
        max_tokens          = kwargs['max_tokens'] if 'max_tokens' in kwargs else instance.compute_remaining_tokens(truncated_prompts_)
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


class InvalidRequestErrorRemedyCompletionStrategy:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, error, prompts_, callback, instance, *args, **kwargs):
        # send prompt to GPT-X Completion-based
        stop                = kwargs['stop'] if 'stop' in kwargs else None
        model               = kwargs['model'] if 'model' in kwargs else None

        msg = str(error)
        handle = None
        try:
            if "This model's maximum context length is" in msg:
                handle = 'type1'
                max_ = instance.max_tokens
                usr = msg.split('tokens. ')[1].split(' ')[-1]
                overflow_tokens = int(usr) - int(max_)
            elif "is less than the minimum" in msg:
                handle = 'type2'
                # extract number until 'is'
                msg_ = msg.split(' ')[0]
                overflow_tokens = int(msg_) * (-1)
            else:
                raise Exception(msg)
        except Exception as e:
            raise e

        # unify the format to use same remedy strategy for both chat and completion
        values = prompts_[0].replace('---------SYSTEM BEHAVIOR--------\n', '').split('\n\n---------USER REQUEST--------\n')
        prompts_ = [{'role': 'system', 'content': values[0]}, {'role': 'user', 'content': values[1]}]

        prompts = [p for p in prompts_ if p['role'] == 'user']
        system_prompt = [p for p in prompts_ if p['role'] == 'system']

        def compute_required_tokens(prompts: dict) -> int:
            # iterate over prompts and compute number of tokens
            prompts_ = [role['content'] for role in prompts]
            prompt = ''.join(prompts_)
            val = len(instance.tokenizer.encode(prompt, disallowed_special=()))
            return val

        def compute_remaining_tokens(prompts: list) -> int:
            val = compute_required_tokens(prompts)
            return int((instance.max_tokens - val) * 0.99)

        if handle == 'type1':
            truncated_content_ = [p['content'][overflow_tokens:] for p in prompts]
            removed_content_ = [p['content'][:overflow_tokens] for p in prompts]
            truncated_prompts_ = [{'role': p['role'], 'content': c} for p, c in zip(prompts, truncated_content_)]
            with ConsoleStyle('warn') as console:
                console.print(f"WARNING: Overflow tokens detected. Reducing prompt size by {overflow_tokens} characters.")
        elif handle == 'type2':
            user_prompts = [p['content'] for p in prompts]
            # truncate until tokens are less than max_tokens * 0.70
            new_prompt = [*system_prompt]
            new_prompt.extend([{'role': p['role'], 'content': c} for p, c in zip(prompts, user_prompts)])

            while compute_required_tokens(new_prompt) > instance.max_tokens * 0.70: # magic number
                user_prompts = [c[overflow_tokens:] for c in user_prompts]
                new_prompt = [*system_prompt]
                new_prompt.extend([{'role': p['role'], 'content': c} for p, c in zip(prompts, user_prompts)])

            with ConsoleStyle('warn') as console:
                console.print(f"WARNING: Overflow tokens detected. Reducing prompt size to {70}% of max_tokens.")
            truncated_prompts_ = [{'role': p['role'], 'content': c} for p, c in zip(prompts, user_prompts)]
        else:
            raise Exception('Invalid handle case for remedy strategy.')

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
        return callback(model=model,
                        prompt=truncated_prompts_,
                        suffix=suffix,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        top_p=top_p,
                        stop=stop,
                        n=1)
