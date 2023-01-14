from .settings import BOTDYN_CONFIG
from typing import List
from .base import Engine
try:
    import whisper
except:
    whisper = None


class WhisperEngine(Engine):
    def __init__(self):
        super().__init__()
        self.model = None # lazy loading
        config = BOTDYN_CONFIG
        self.model_id = config['SPEECH_ENGINE_MODEL']

    def forward(self, *args, **kwargs) -> List[str]:
        assert whisper is not None, "Whisper is not installed. Please install it first."
        if self.model is None:
            self.model = whisper.load_model(self.model_id)
        prompt = kwargs['prompt']
        audio = kwargs['audio']
        
        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((prompt, audio))
        
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        if prompt == 'detect_language':
            # detect the spoken language
            _, probs = self.model.detect_language(mel)
            rsp = max(probs, key=probs.get)
        elif prompt == 'decode':
            # decode the audio
            options = whisper.DecodingOptions(fp16 = False)
            result = whisper.decode(self.model, mel, options)
            rsp = result.text
        else:
            raise Exception(f"Unknown whisper command prompt: {prompt}")
        
        output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
        if output_handler:
            output_handler(rsp)
        
        return [rsp]
    
    def prepare(self, args, kwargs, wrp_params):
        assert 'audio' in wrp_params, "Whisper requires audio input."
        audio_file = wrp_params['audio']
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(audio_file)
        audio = whisper.pad_or_trim(audio)
        wrp_params['audio'] = audio
    