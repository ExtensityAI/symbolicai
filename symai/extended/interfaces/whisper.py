from ... import core
from ...backend.engines.speech_to_text.engine_local_whisper import WhisperResult
from ...symbol import Expression


class whisper(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__

    def __call__(self, audio_path: str, operation: str = 'decode', **kwargs) -> WhisperResult:
        @core.speech_to_text(audio=audio_path, prompt=operation, **kwargs)
        def _func(_) -> WhisperResult:
            pass
        return _func(self)
