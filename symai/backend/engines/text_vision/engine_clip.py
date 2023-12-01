import logging
import requests

from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ...base import Engine
from ...settings import SYMAI_CONFIG


# supress warnings
logging.getLogger("PIL").setLevel(logging.WARNING)


class CLIPEngine(Engine):
    def __init__(self):
        super().__init__()
        self.model        =  None # lazy loading
        self.preprocessor = None # lazy loading

        self.config       = SYMAI_CONFIG
        self.model_id     = self.config['VISION_ENGINE_MODEL']
        self.old_model_id = self.config['VISION_ENGINE_MODEL']

    def id(self) -> str:
        if self.config['VISION_ENGINE_MODEL']:
            return 'text_vision'
        return super().id() # default to unregistered

    def command(self, argument):
        super().command(argument.kwargs)
        if 'VISION_ENGINE_MODEL' in argument.kwargs:
            self.model_id     = argument.kwargs['VISION_ENGINE_MODEL']

    def forward(self, argument):
        image_url, text       = argument.prop.prepared_input
        kwargs                = argument.kwargs

        if self.model is None or self.model_id != self.old_model_id:
            self.model        = CLIPModel.from_pretrained(self.model_id)
            self.processor    = CLIPProcessor.from_pretrained(self.model_id)
            self.old_model_id = self.model_id

        if text is None:
            image             = Image.open(requests.get(image_url, stream=True).raw)
            inputs            = self.processor(images=image, return_tensors="pt")
            rsp               = self.model.get_image_features(**inputs)
        elif image_url is None:
            inputs            = self.processor(text=text, return_tensors="pt")
            rsp               = self.model.get_text_features(**inputs)
        else:
            image             = Image.open(requests.get(image_url, stream=True).raw)
            inputs            = self.processor(text=text, images=image, return_tensors="pt", padding=True)
            outputs           = self.model(**inputs)
            logits_per_image  = outputs.logits_per_image  # this is the image-text similarity score
            rsp               = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

        rsp = rsp.detach().cpu().numpy()

        metadata = {}

        return [rsp], metadata

    def prepare(self, argument):
        assert not argument.prop.processed_input, "CLIPEngine does not support processed_input."
        kwargs     = argument.kwargs
        image_url  = str(argument.kwargs['image']) if 'image' in kwargs else None
        text       = str(argument.kwargs['text'])  if 'text'  in kwargs else None
        argument.prop.prepared_input = (image_url, text)

