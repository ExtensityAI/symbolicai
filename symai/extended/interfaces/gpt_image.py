from ... import core
from ...backend.engines.drawing.engine_gpt_image import GPTImageResult
from ...symbol import Expression, Symbol


class gpt_image(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__

    def __call__(self, prompt: str | None = None, operation: str = 'create', engine: str | None = None, model: str | None = None, **kwargs) -> GPTImageResult:
            prompt = self._to_symbol(prompt)
            @core.draw(engine=engine or "drawing", operation=operation, model=model, **kwargs)
            def _func(_) -> GPTImageResult:
                pass
            return _func(prompt)
