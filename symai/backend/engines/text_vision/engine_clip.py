import logging
from io import BytesIO

import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ....utils import UserMessage
from ...base import Engine
from ...settings import SYMAI_CONFIG

# supress warnings
logging.getLogger("PIL").setLevel(logging.WARNING)


class CLIPEngine(Engine):
    def __init__(self, model: str | None = None):
        super().__init__()
        self.model =  None # lazy loading
        self.preprocessor = None # lazy loading
        self.config = SYMAI_CONFIG
        self.model_id = self.config['VISION_ENGINE_MODEL'] if model is None else model
        self.old_model_id = self.config['VISION_ENGINE_MODEL'] if model is None else model
        self.name = self.__class__.__name__

    def id(self) -> str:
        if self.config['VISION_ENGINE_MODEL']:
            return 'text_vision'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'VISION_ENGINE_MODEL' in kwargs:
            self.model_id     = kwargs['VISION_ENGINE_MODEL']

    def load_images(self, image):
        images = []
        if not isinstance(image, (list, tuple)):
            image = [image]

        for img in image:
            if isinstance(img, bytes):
                images.append(Image.open(BytesIO(img)))
            elif isinstance(img, str):
                image_source = requests.get(img, stream=True).raw if img.startswith('http') else img
                image = Image.open(image_source)
                images.append(image)
        return images

    def forward(self, argument):
        image_url, text       = argument.prop.prepared_input

        if self.model is None or self.model_id != self.old_model_id:
            self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model        = CLIPModel.from_pretrained(self.model_id).to(self.device)
            self.processor    = CLIPProcessor.from_pretrained(self.model_id)
            self.old_model_id = self.model_id

        if text is None and image_url is not None:
            image             = self.load_images(image_url)
            inputs            = self.processor(images=image, return_tensors="pt").to(self.device)
            rsp               = self.model.get_image_features(**inputs)
        elif image_url is None and text is not None:
            inputs            = self.processor(text=text, return_tensors="pt").to(self.device)
            rsp               = self.model.get_text_features(**inputs)
        elif image_url is not None and text is not None:
            image             = self.load_images(image_url)
            inputs            = self.processor(text=text, images=image, return_tensors="pt", padding=True).to(self.device)
            outputs           = self.model(**inputs)
            logits_per_image  = outputs.logits_per_image  # this is the image-text similarity score
            rsp               = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        else:
            UserMessage("CLIPEngine requires either image or text input.", raise_with=NotImplementedError)

        rsp = rsp.squeeze().detach().cpu().numpy()

        metadata = {}

        return [rsp], metadata

    def prepare(self, argument):
        assert not argument.prop.processed_input, "CLIPEngine does not support processed_input."
        kwargs     = argument.kwargs
        image_url  = argument.kwargs['image'] if 'image' in kwargs else None
        text       = argument.kwargs['text']  if 'text'  in kwargs else None
        argument.prop.prepared_input = (image_url, text)
