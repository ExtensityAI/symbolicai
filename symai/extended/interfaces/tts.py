from ... import core
from ...symbol import Expression, Symbol


class tts(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, prompt: Symbol, path: str, voice: str = 'nova', **kwargs) -> "tts":
        prompt = str(prompt)
        @core.text_to_speech(prompt=prompt, path=path, voice=voice, **kwargs)
        def _func(_) -> str:
            pass
        return self.sym_return_type(_func(self))
