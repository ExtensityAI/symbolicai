import logging
from typing import Iterable, List

import torch
from tqdm import tqdm

from ..formatter import WhisperTimestampsFormatter
from .base import Engine
from .settings import SYMAI_CONFIG

try:
    import whisper
    from whisper.audio import \
        N_SAMPLES  # @NOTE: sample_rate (16_000) * chunk_length (30) = 480_000
    from whisper.tokenizer import get_tokenizer
except ImportError:
    whisper = None
    N_SAMPLES = 16_000 * 30


class WhisperEngine(Engine):
    def __init__(self):
        super().__init__()
        config = SYMAI_CONFIG
        self.model = None # lazy loading
        self.model_id = config['SPEECH_ENGINE_MODEL']
        self.old_model_id = config['SPEECH_ENGINE_MODEL']
        self.tokens = []
        self.text = []
        self.formatter = WhisperTimestampsFormatter()

    def command(self, wrp_params):
        super().command(wrp_params)
        if 'SPEECH_ENGINE_MODEL' in wrp_params:
            self.model_id = wrp_params['SPEECH_ENGINE_MODEL']

    def forward(self, **kwargs) -> List[str]:
        assert whisper is not None, "Whisper is not installed. Please install it first."
        if self.model is None or self.model_id != self.old_model_id:
            device_fallback = 'cpu'
            device = "cuda" if torch.cuda.is_available() else device_fallback
            device = kwargs['device'] if 'device' in kwargs else device # user preference over auto detection
            try:
                self.model = whisper.load_model(self.model_id, device=device)
            except RuntimeError:
                logging.warn(f"Whisper failed to load model on device {device}. Fallback to {device_fallback}.")
                self.model = whisper.load_model(self.model_id, device=device_fallback)
            self.old_model_id = self.model_id

        self._try_compile()
        prompt = kwargs['prompt']
        audio  = kwargs['audio']
        disable_pbar    = kwargs.get("disable_pbar", False)
        language        = kwargs.get("language", "en")
        temperature     = kwargs.get("temperature", (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
        word_timestamps = kwargs.get("word_timestamps", False)
        input_handler   = kwargs.get("input_handler")
        if input_handler is not None:
            input_handler((prompt, audio))

        if prompt == 'detect_language':
            #@NOTE: the accuracy of mel spectrogram is not good enough; don't use it to transcribe
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            _, probs = self.model.detect_language(mel)
            rsp = max(probs, key=probs.get)
        elif prompt == 'decode':
            pbar = tqdm(self._get_chunks(audio), disable=disable_pbar)
            for chunk in pbar:
                result = self.model.transcribe(
                    chunk,
                    language=language,
                    word_timestamps=word_timestamps,
                    temperature=temperature,
                    fp16=False,
                )
                self.text.append(result["text"])
                self.tokens.append([
                    token
                    for segment in result["segments"]
                    for token in segment["tokens"]
                ])
            if word_timestamps is not None:
                tokenizer = get_tokenizer(self.model.is_multilingual)
                tokens = [tokenizer.decode_with_timestamps(t) for t in self.tokens]
                rsp = self.formatter(tokens)
            else:
                rsp = " ".join(self.text)
        else:
            raise Exception(f"Unknown whisper command prompt: {prompt}")

        output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
        if output_handler:
            output_handler(rsp)

        metadata = {}
        if 'metadata' in kwargs and kwargs['metadata']:
            metadata['kwargs'] = kwargs
            metadata['input']  = (prompt, audio)
            metadata['output'] = rsp

        return [rsp], metadata

    def prepare(self, args, kwargs, wrp_params):
        assert 'audio' in wrp_params, "Whisper requires audio input."
        audio_file = str(wrp_params['audio'])
        audio = whisper.load_audio(audio_file)
        wrp_params['audio'] = audio

    def _get_chunks(self, it: Iterable, batch: int = N_SAMPLES) -> torch.Tensor:
        """
        Split an iterable into chunks of size `batch`. It defaults to `N_SAMPLES` 480_000 samples which is equal to 30 seconds.
        """
        size = len(it)
        for i in range(0, size, batch):
            yield torch.tensor(it[i:min(i + batch, size)]).to(self.model.device)

    def _try_compile(self):
        try:
            self.model = torch.compile(self.model)
        except Exception:
            pass
