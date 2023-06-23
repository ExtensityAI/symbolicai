from ...symbol import Expression
from ... import core


class whisper(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, audio_path: str, operation: str = 'decode', **kwargs) -> "whisper":
        @core.speech(audio=audio_path, prompt=operation, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
