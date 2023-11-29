from ... import core
from ...symbol import Expression
from ...backend.engines.ocr.engine_apilayer import ApiLayerResult


class ocr(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, image_url: str, **kwargs) -> ApiLayerResult:
        if not image_url.startswith('http'):
            image_url = f'file://{image_url}'
        @core.ocr(image=image_url, **kwargs)
        def _func(_) -> ApiLayerResult:
            pass
        return _func(self)
