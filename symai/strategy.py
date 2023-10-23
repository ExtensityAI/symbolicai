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


class InvalidRequestErrorRemedyStrategy:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, error, prompts_, callback, instance, *args, **kwargs):
        # send prompt to GPT-X Chat-based
        stop                = kwargs['stop'] if 'stop' in kwargs else None
        model               = kwargs['model'] if 'model' in kwargs else None

        msg = str(error)

        tolerance = 10
        if "This model's maximum context length is" in msg:
            usr = msg.split('(')[-1].split(' ')[0]
            max_ = msg.split('tokens')[0].split(' ')[-1]
            overflow_tokens = int(usr) - int(max_) + tolerance
        else:
            # extract number until 'is'
            overflow_tokens = int(msg.split(' ')[0]) * (-1) + tolerance

        prompts = [p for p in prompts_ if p['role'] == 'user']
        truncated_content_ = [p['content'][overflow_tokens:] for p in prompts]
        removed_content_ = [p['content'][:overflow_tokens] for p in prompts]
        truncated_prompts_ = [{'role': p['role'], 'content': c} for p, c in zip(prompts, truncated_content_)]
        with ConsoleStyle('warn') as console:
            console.print(f"WARNING: Overflow tokens detected. Reducing prompt size by {overflow_tokens} characters.")

        # convert map to list of strings
        max_tokens          = kwargs['max_tokens'] if 'max_tokens' in kwargs else instance.compute_remaining_tokens(truncated_prompts_)
        temperature         = kwargs['temperature'] if 'temperature' in kwargs else 1
        frequency_penalty   = kwargs['frequency_penalty'] if 'frequency_penalty' in kwargs else 0
        presence_penalty    = kwargs['presence_penalty'] if 'presence_penalty' in kwargs else 0
        top_p               = kwargs['top_p'] if 'top_p' in kwargs else 1

        return callback(model=model,
                        messages=truncated_prompts_,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        top_p=top_p,
                        stop=stop,
                        n=1)
