from ... import core
from ...backend.engines.drawing.engine_gemini_image import GeminiImageResult
from ...symbol import Expression


class nanobanana(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__

    def __call__(
        self,
        prompt: str | None = None,
        operation: str = "create",
        **kwargs,
    ) -> list:
        prompt = self._to_symbol(prompt)

        @core.draw(operation=operation, **kwargs)
        def _func(_) -> GeminiImageResult:
            pass

        return _func(prompt).value
