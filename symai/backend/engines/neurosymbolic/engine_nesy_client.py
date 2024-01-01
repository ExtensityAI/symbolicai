import logging
import rpyc

from typing import List

from ...base import Engine
from ...settings import SYMAI_CONFIG


rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True


class NeSyClientEngine(Engine):
    def __init__(self, host: str = 'localhost', port: int = 18100, timeout: int = 240):
        super().__init__()
        logger              = logging.getLogger('nesy_client')
        logger.setLevel(logging.WARNING)
        self. config        = SYMAI_CONFIG
        self.host           = host
        self.port           = port
        self.timeout        = timeout
        self.connection     = None
        self.seed           = None
        self.except_remedy  = None

    def id(self) -> str:
        if  self.config['NEUROSYMBOLIC_ENGINE_MODEL'] and \
            self.config['NEUROSYMBOLIC_ENGINE_MODEL'] == 'localhost':
            return 'neurosymbolic'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'seed' in kwargs:
            self.seed      = kwargs['seed']
        if 'except_remedy' in kwargs:
            self.except_remedy = kwargs['except_remedy']

    @property
    def max_tokens(self):
        return self.connection.root.max_tokens()

    def forward(self, argument):
        if self.connection is None:
            self.connection = rpyc.connect(self.host, self.port)
            self.connection._config['sync_request_timeout'] = self.timeout

        prompts       = argument.prop.prepared_input
        kwargs        = argument.kwargs
        prompts_      = prompts if isinstance(prompts, list) else [prompts]

        # send prompt to Model
        stop          = kwargs['stop'] if 'stop' in kwargs else None
        suffix        = kwargs['template_suffix'] if 'template_suffix' in kwargs else None
        max_tokens    = kwargs['max_tokens'] if 'max_tokens' in kwargs else 128
        temperature   = kwargs['temperature'] if 'temperature' in kwargs else 0.7
        top_p         = kwargs['top_p'] if 'top_p' in kwargs else 0.95
        top_k         = kwargs['top_k'] if 'top_k' in kwargs else 50
        seed          = kwargs['seed'] if 'seed' in kwargs else self.seed
        except_remedy = kwargs['except_remedy'] if 'except_remedy' in kwargs else self.except_remedy

        config = {}
        # add to config only if not None
        if stop is not None:
            config['stop_words'] = stop
        if suffix is not None:
            config['suffix'] = suffix
        if max_tokens is not None:
            config['max_new_tokens'] = max_tokens
        if temperature is not None:
            config['temperature'] = temperature
        if top_p is not None:
            config['top_p'] = top_p
        if top_k is not None:
            config['top_k'] = top_k
        if seed is not None:
            config['seed'] = seed

        try:
            # use RPyC to send prompt to Model
            res = self.connection.root.predict(prompts_[0], **config)
        except Exception as e:
            if except_remedy is None:
                raise e
            callback = self.connection.root.predict
            res = except_remedy(self, e, callback, argument)

        metadata = {}

        rsp    = [res]
        output = rsp if isinstance(prompts, list) else rsp[0]
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

        user:   str = ""
        system: str = ""
        system      = f'{system}\n' if system and len(system) > 0 else ''

        ref = argument.prop.instance
        static_ctxt, dyn_ctxt = ref.global_context
        if len(static_ctxt) > 0:
            system += f"[STATIC CONTEXT]\n{static_ctxt}\n\n"

        if len(dyn_ctxt) > 0:
            system += f"[DYNAMIC CONTEXT]\n{dyn_ctxt}\n\n"

        payload = argument.prop.payload
        if payload is not None:
            system += f"[ADDITIONAL CONTEXT]\n{str(payload)}\n\n"

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
            user += f"\n[[PLACEHOLDER]]\n{str(argument.prop.template_suffix)}\n\n"
            user += f"Only generate content for the placeholder `[[PLACEHOLDER]]` following the instructions and context information. Do NOT write `[[PLACEHOLDER]]` or anything else in your output.\n\n"

        argument.prop.prepared_input = [f'---------SYSTEM BEHAVIOR--------\n{system}\n\n---------USER REQUEST--------\n{user}']
