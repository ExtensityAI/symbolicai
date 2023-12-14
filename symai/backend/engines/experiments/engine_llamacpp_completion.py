import logging
from typing import List

from ...base import Engine
from ...settings import SYMAI_CONFIG

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

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'NEUROSYMBOLIC_ENGINE_MODEL' in kwargs:
            self.model = kwargs['NEUROSYMBOLIC_ENGINE_MODEL']

    def forward(self, argument):
        prompts_            = argument.prop.prepared_input
        prompts_            = prompts_ if isinstance(prompts_, list) else [prompts_]
        kwargs              = argument.kwargs
        max_tokens          = kwargs['max_tokens'] if 'max_tokens' in kwargs else 4096
        temperature         = kwargs['temperature'] if 'temperature' in kwargs else 0.7
        top_p               = kwargs['top_p'] if 'top_p' in kwargs else 1
        except_remedy       = kwargs['except_remedy'] if 'except_remedy' in kwargs else None

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
        except Exception as e:
            if except_remedy is None:
                raise e
            callback = self.connection.root.predict
            res = except_remedy(self, e, callback, argument)


        metadata = {}

        rsp    = [r['text'] for r in res['choices']]
        output = rsp if isinstance(prompts_, list) else rsp[0]
        return output, metadata

    def prepare(self, argument):
        if argument.prop.raw_input:
            if not argument.prop.processed_input:
                raise ValueError('Need to provide a prompt instruction to the engine if raw_input is enabled.')
            argument.prop.prepared_input = argument.prop.processed_input
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


if __name__ == '__main__':
    engine = LLaMACppCompletionEngine()
    res = engine.forward(["\n\n### Instructions:\nWhat is the capital of Romania?\n\n### Response:\n"])
    print(res)