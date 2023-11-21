import logging
from typing import List

import rpyc

from .base import Engine

rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True


class NeSyClientEngine(Engine):
    def __init__(self, host: str = 'localhost', port: int = 18100, timeout: int = 240):
        super().__init__()
        logger = logging.getLogger('nesy_client')
        logger.setLevel(logging.WARNING)
        self.host       = host
        self.port       = port
        self.timeout    = timeout
        self.connection = None

    @property
    def max_tokens(self):
        return self.connection.root.max_tokens()

    def forward(self, prompts: List[str], *args, **kwargs) -> List[str]:
        if self.connection is None:
            self.connection = rpyc.connect(self.host, self.port)
            self.connection._config['sync_request_timeout'] = self.timeout

        prompts_      = prompts if isinstance(prompts, list) else [prompts]
        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((prompts_,))

        # send prompt to Model
        stop          = kwargs['stop'] if 'stop' in kwargs else None
        suffix        = kwargs['template_suffix'] if 'template_suffix' in kwargs else None
        max_tokens    = kwargs['max_tokens'] if 'max_tokens' in kwargs else 128
        temperature   = kwargs['temperature'] if 'temperature' in kwargs else 0.7
        top_p         = kwargs['top_p'] if 'top_p' in kwargs else 0.95
        top_k         = kwargs['top_k'] if 'top_k' in kwargs else 50
        except_remedy = kwargs['except_remedy'] if 'except_remedy' in kwargs else None

        config = {
        }
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

        try:
            # use RPyC to send prompt to Model
            res = self.connection.root.predict(prompts_[0], **config)
            output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
            if output_handler:
                output_handler(res)
        except Exception as e:
            if except_remedy is None:
                raise e
            callback = self.connection.root.predict
            res = except_remedy(e, prompts_, callback, self, *args, **kwargs)

        metadata = {}
        if 'metadata' in kwargs and kwargs['metadata']:
            metadata['kwargs'] = kwargs
            metadata['input']  = prompts_
            metadata['output'] = res

        rsp    = [res]
        output = rsp if isinstance(prompts, list) else rsp[0]
        return output, metadata

    def prepare(self, args, kwargs, wrp_params):
        if 'raw_input' in wrp_params:
            wrp_params['prompts'] = wrp_params['raw_input']
            return

        user:   str = ""
        system: str = ""
        system      = f'{system}\n' if system and len(system) > 0 else ''

        ref = wrp_params['wrp_self']
        static_ctxt, dyn_ctxt = ref.global_context
        if len(static_ctxt) > 0:
            system += f"[STATIC CONTEXT]\n{static_ctxt}\n\n"

        if len(dyn_ctxt) > 0:
            system += f"[DYNAMIC CONTEXT]\n{dyn_ctxt}\n\n"

        payload = wrp_params['payload'] if 'payload' in wrp_params else None
        if payload is not None:
            system += f"[ADDITIONAL CONTEXT]\n{payload}\n\n"

        examples: List[str] = wrp_params['examples']
        if examples and len(examples) > 0:
            system += f"[EXAMPLES]\n{str(examples)}\n\n"

        if wrp_params['prompt'] is not None and len(wrp_params['prompt']) > 0 and ']: <<<' not in str(wrp_params['processed_input']): # TODO: fix chat hack
            user += f"[INSTRUCTION]\n{str(wrp_params['prompt'])}"

        suffix: str = wrp_params['processed_input']
        if '=>' in suffix:
            user += f"[LAST TASK]\n"

        parse_system_instructions = False if 'parse_system_instructions' not in wrp_params else wrp_params['parse_system_instructions']
        if '[SYSTEM_INSTRUCTION::]: <<<' in suffix and parse_system_instructions:
            parts = suffix.split('\n>>>\n')
            # first parts are the system instructions
            for p in parts[:-1]:
                system += f"{p}\n"
            # last part is the user input
            suffix = parts[-1]
        user += f"{suffix}"

        template_suffix = wrp_params['template_suffix'] if 'template_suffix' in wrp_params else None
        if template_suffix:
            user += f"\n[[PLACEHOLDER]]\n{template_suffix}\n\n"
            user += f"Only generate content for the placeholder `[[PLACEHOLDER]]` following the instructions and context information. Do NOT write `[[PLACEHOLDER]]` or anything else in your output.\n\n"

        wrp_params['prompts'] = [f'---------SYSTEM BEHAVIOR--------\n{system}\n\n---------USER REQUEST--------\n{user}']
