from ... import core
from ...symbol import Expression, Symbol, Result


class tts(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, prompt: Symbol, path: str, voice: str = 'nova', **kwargs) -> Result:
        @core.text_to_speech(prompt=str(prompt), path=path, voice=voice, **kwargs)
        def _func(_) -> Result:
            pass
        return _func(self)
