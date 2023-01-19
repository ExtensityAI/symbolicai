from typing import List
from .base import Engine
from .settings import SYMAI_CONFIG
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel


class CLIPEngine(Engine):
    def __init__(self):
        super().__init__()
        self.model =  None # lazy loading
        self.preprocessor = None # lazy loading
        
        config = SYMAI_CONFIG
        self.model_id = config['VISION_ENGINE_MODEL']

    def forward(self, *args, **kwargs) -> List[str]:        
        if self.model is None:
            self.model = CLIPModel.from_pretrained(self.model_id)
            self.processor = CLIPProcessor.from_pretrained(self.model_id)
        
        image_url = kwargs['image'] if 'image' in kwargs else None
        text = kwargs['prompt']
        
        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((image_url, text))
        
        if text is None:
            image = Image.open(requests.get(image_url, stream=True).raw)
            inputs = self.processor(images=image, return_tensors="pt")
            rsp = self.model.get_image_features(**inputs)
        elif image_url is None:
            inputs = self.processor(text=text, return_tensors="pt")
            rsp = self.model.get_text_features(**inputs)
        else:
            image = Image.open(requests.get(image_url, stream=True).raw)
            inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            rsp = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        
        output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
        if output_handler:
            output_handler(rsp)
        
        rsp = rsp.detach().cpu().numpy()
        return [rsp]
    
    def prepare(self, args, kwargs, wrp_params):
        pass
    