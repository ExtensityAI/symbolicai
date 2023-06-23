import logging
from typing import List

import rpyc

from .base import Engine


class NeSyClientEngine(Engine):
    def __init__(self, host: str = 'localhost', port: int = 18100, timeout: int = 240):
        super().__init__()
        logger = logging.getLogger('nesy_client')
        logger.setLevel(logging.WARNING)
        self.host = host
        self.port = port
        self.timeout = timeout
        self.connection = None

    def forward(self, prompts: List[str], *args, **kwargs) -> List[str]:
        if self.connection is None:
            self.connection = rpyc.connect(self.host, self.port)
            self.connection._config['sync_request_timeout'] = self.timeout

        prompts_ = prompts if isinstance(prompts, list) else [prompts]
        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((prompts_,))

        # send prompt to Model
        stop            = kwargs['stop'] if 'stop' in kwargs else None
        suffix          = kwargs['template_suffix'] if 'template_suffix' in kwargs else None
        max_tokens      = kwargs['max_tokens'] if 'max_tokens' in kwargs else 128
        temperature     = kwargs['temperature'] if 'temperature' in kwargs else 0.7
        top_p           = kwargs['top_p'] if 'top_p' in kwargs else 0.95
        top_k           = kwargs['top_k'] if 'top_k' in kwargs else 50
        except_remedy   = kwargs['except_remedy'] if 'except_remedy' in kwargs else None

        try:
            # use RPyC to send prompt to Model
            res = self.connection.root.predict(prompts_[0],
                                                max_tokens=max_tokens,
                                                temperature=temperature,
                                                top_p=top_p,
                                                top_k=top_k,
                                                stop=stop,
                                                suffix=suffix)
            output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
            if output_handler:
                output_handler(res)
        except Exception as e:
            if except_remedy is None:
                raise e
            rsp = except_remedy(e, prompts_, *args, **kwargs)

        rsp = [res]
        return rsp if isinstance(prompts, list) else rsp[0]

    def prepare(self, args, kwargs, wrp_params):
        prompt: str = ''
        # add static context
        ref = wrp_params['wrp_self']
        static_ctxt, dyn_ctxt = ref.global_context
        if len(static_ctxt) > 0:
            prompt += f"General Context:\n{static_ctxt}\n\n----------------\n\n"
        if wrp_params['prompt'] is not None: # TODO: check if this works
            prompt += str(wrp_params['prompt'])
        # build operation
        message = f'{prompt}\n' if prompt and len(prompt) > 0 else ''
        # add examples
        examples: List[str] = wrp_params['examples']
        if examples:
            message += f"Examples:\n"
            message += f"{str(examples)}\n"
        # add dynamic context
        if len(dyn_ctxt) > 0:
            message += f"\n\n----------------\n\nDynamic Context:\n{dyn_ctxt}"
        # add method payload
        payload = wrp_params['payload'] if 'payload' in wrp_params else None
        if payload is not None:
            message += f"\n\n----------------\n\nAdditional Context: {payload}"

        # add user request
        suffix: str = wrp_params['processed_input']
        if '=>' in suffix:
            message += f"Last Task:\n"
            message += f"----------------\n\n"
        message += f"{suffix}"
        wrp_params['prompts'] = [message]
