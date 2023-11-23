import logging
from typing import List

from .base import Engine
from .settings import SYMAI_CONFIG

import requests
import json


class LLaMACppCompletionClient():
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def create_completion(self, completion_request):
        # Preparing header information
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        # Performing POST request
        response = requests.post(f"{self.base_url}/v1/engines/copilot-codex/completions",
                                 data=json.dumps(completion_request),
                                 headers=headers)

        print(response)

        # Parsing response
        if response.status_code == 200:
            return json.loads(response.text)
        else:
            raise Exception(f"Request failed with status code {response.status_code}")


class LLaMACppCompletionEngine(Engine):
    def __init__(self):
        super().__init__()
        config          = SYMAI_CONFIG
        self.model      = config['NEUROSYMBOLIC_ENGINE_MODEL']
        logger          = logging.getLogger('LLaMA')
        logger.setLevel(logging.WARNING)

        self.client = LLaMACppCompletionClient() # Initialize LLaMACpp client here

    def command(self, wrp_params):
        super().command(wrp_params)
        if 'NEUROSYMBOLIC_ENGINE_MODEL' in wrp_params:
            self.model = wrp_params['NEUROSYMBOLIC_ENGINE_MODEL']

    def forward(self, prompts: List[str], *args, **kwargs) -> List[str]:
        prompts_            = prompts if isinstance(prompts, list) else [prompts]
        max_tokens          = kwargs['max_tokens'] if 'max_tokens' in kwargs else 4096
        temperature         = kwargs['temperature'] if 'temperature' in kwargs else 0.7
        top_p               = kwargs['top_p'] if 'top_p' in kwargs else 1

        completion_request = {
            "prompt": prompts_,
            "stop": ["\n\n", "-------"]
            #"max_tokens": max_tokens,
            #"temperature": temperature,
            #"top_p": top_p,
            #"mirostat_mode": 0,
            #"mirostat_tau": 0.2,
            #"mirostat_eta": 0.5,
            #"echo": False
        }

        try:
            print(completion_request)
            res = self.client.create_completion(completion_request)
            output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
            if output_handler:
                output_handler(res)
        except Exception as e:

            if kwargs.get('except_remedy'):
                res = kwargs['except_remedy'](e, prompts_, self.client.create_completion, self, *args, **kwargs)
            else:
                raise e

        metadata = {}
        if 'metadata' in kwargs and kwargs['metadata']:
            metadata['kwargs']      = kwargs
            metadata['input']       = prompts_
            metadata['output']      = res
            metadata['model']       = self.model
            metadata['max_tokens']  = max_tokens
            metadata['temperature'] = temperature
            metadata['top_p']       = top_p

        rsp    = [r['text'] for r in res['choices']]
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

        wrp_params['prompts'] = [f'---------SYSTEM BEHAVIOR--------\n{system}\n\n---------USER REQUEST--------\n{user}\n\n[RESPONSE]\n]']


if __name__ == '__main__':
    engine = LLaMACppCompletionEngine()
    res = engine.forward(["\n\n### Instructions:\nWhat is the capital of Romania?\n\n### Response:\n"])
    print(res)