from .settings import SYMAI_CONFIG
from typing import List
from .base import Engine
from time import sleep
import openai
import logging


class ImageRenderingEngine(Engine):
    def __init__(self, max_retry: int = 3, api_cooldown_delay: int = 3, size: int = 512):
        super().__init__()
        config = SYMAI_CONFIG
        openai.api_key = config['IMAGERENDERING_ENGINE_API_KEY']
        logger = logging.getLogger('openai')
        logger.setLevel(logging.WARNING)
        self.max_retry = max_retry
        self.api_cooldown_delay = api_cooldown_delay
        self.size = size

    def forward(self, prompt: str, *args, **kwargs) -> List[str]:
        retry: int = 0
        success: bool = False
        errors: List[Exception] = []
        
        size = f"{kwargs['image_size']}x{kwargs['image_size']}" if 'image_size' in kwargs else f"{self.size}x{self.size}"
        
        max_retry = kwargs['max_retry'] if 'max_retry' in kwargs else self.max_retry
        while not success and retry < max_retry:
            try:
                if kwargs['operation'] == 'create':
                    input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
                    if input_handler:
                        input_handler((prompt,))
                    
                    res = openai.Image.create(
                        prompt=prompt,
                        n=1,
                        size=size
                    )
                elif kwargs['operation'] == 'variation':
                    assert 'image_path' in kwargs
                    image_path = kwargs['image_path']
                    
                    input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
                    if input_handler:
                        input_handler((prompt, image_path))
                    
                    res = openai.Image.create_variation(
                        image=open(image_path, "rb"),
                        n=1,
                        size=size
                    )
                elif kwargs['operation'] == 'edit':
                    assert 'mask_path' in kwargs
                    assert 'image_path' in kwargs
                    mask_path = kwargs['mask_path']
                    image_path = kwargs['image_path']
                    
                    input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
                    if input_handler:
                        input_handler((prompt, image_path, mask_path))
                    
                    res = openai.Image.create_edit(
                        image=open(image_path, "rb"),
                        mask=open(mask_path, "rb"),
                        prompt=prompt,
                        n=1,
                        size=size
                    )
                else:
                    raise Exception(f"Unknown operation: {kwargs['operation']}")
                
                output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
                if output_handler:
                    output_handler(res)
                
                success = True
            except Exception as e:
                errors.append(e)
                self.logger.warn(f"DALL-E service is unavailable or caused an error. Retry triggered: {e}")
                sleep(self.api_cooldown_delay) # API cooldown
            retry += 1
        
        if not success:
            msg = f"Failed to query DALL-E after {max_retry} retries. Errors: {errors}"
            # interpret error
            from symai.symbol import Symbol
            from symai.components import Analyze
            sym = Symbol(errors)
            expr = Analyze(exception=errors[-1], query="Explain the issue in this error message")
            sym.stream(expr=expr, max_retry=1)
            msg_reply = f"{msg}\n Analysis: {sym}"
            raise Exception(msg_reply)
        
        rsp = res['data'][0]['url']
        return [rsp]
    
    def prepare(self, args, kwargs, wrp_params):
        prompt = wrp_params['prompt']
        prompt += wrp_params['processed_input']
        wrp_params['prompt'] = prompt
