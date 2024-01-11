import logging
import torch

from itertools import takewhile
from typing import Iterable, List, Optional

from ...base import Engine
from ...settings import SYMAI_CONFIG
from ....symbol import Expression, Result


class WhisperTimestampsFormatter(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, response: List[str]) -> str:
        result = []
        for i, interval in enumerate(response):
            interval = self._filter_empty_string(interval)
            prev_end = 0.0
            prev_start = 0.0
            for head, tail in zip(interval[::2], interval[1::2]):
                start = self._get_timestamp(head)
                end = self._get_timestamp(tail)
                if start >= prev_end:
                    start = prev_end
                    prev_end = end
                    prev_start = start
                    result.append(f"{self._format_to_hours(start + (i*30))} {self._get_sentence(head)}")
                    continue
                if start < prev_start:
                    continue
                delta = end - start
                if start + prev_end > 30:
                    start = prev_end
                else:
                    start += prev_end
                if start + delta > 30:
                    end = 30
                else:
                    end = start + delta
                prev_end = end
                result.append(f"{self._format_to_hours(start + (i*30))} {self._get_sentence(head)}")
        return "\n".join(result)

    def _filter_empty_string(self, s: str) -> List[str]:
        return list(filter(lambda x: x, s.split("<|")))

    def _get_timestamp(self, s: str) -> float:
        return float("".join(list(takewhile(lambda x: x != "|", s))))

    def _get_sentence(self, s: str) -> str:
        return s.split("|>")[-1]

    def _format_to_hours(self, seconds: float) -> str:
        hours    = int(seconds // 3600)
        seconds %= 3600
        minutes  = int(seconds // 60)
        seconds %= 60
        formatted_time = "{:02d}:{:02d}:{:02d}".format(hours, minutes, int(seconds))
        return formatted_time


class WhisperResult(Result):
    def __init__(self, value, **kwargs) -> None:
        super().__init__(value, **kwargs)
        self.raw = None


try:
    import whisper
    from whisper.audio import \
        N_SAMPLES  # @NOTE: sample_rate (16_000) * chunk_length (30) = 480_000
    from whisper.tokenizer import get_tokenizer
except ImportError:
    whisper   = None
    N_SAMPLES = 16_000 * 30


class WhisperEngine(Engine):
    def __init__(self, model: Optional[str] = None):
        super().__init__()
        self.config       = SYMAI_CONFIG
        self.model        = None # lazy loading
        self.model_id     = self.config['SPEECH_TO_TEXT_ENGINE_MODEL'] if model is None else model
        self.old_model_id = self.config['SPEECH_TO_TEXT_ENGINE_MODEL'] if model is None else model
        self.tokens       = []
        self.text         = []
        self.formatter    = WhisperTimestampsFormatter()

    def id(self) -> str:
        if self.config['SPEECH_TO_TEXT_ENGINE_MODEL']:
            if whisper is None:
                print("Whisper is not installed. Please install it with `pip install symbolicai[whisper]`")
            return 'speech-to-text'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'SPEECH_TO_TEXT_ENGINE_MODEL' in kwargs:
            self.model_id = kwargs['SPEECH_TO_TEXT_ENGINE_MODEL']

    def forward(self, argument):
        assert whisper is not None, "Whisper is not installed. Please install it first."
        kwargs     = argument.kwargs
        (_, audio) = argument.prop.prepared_input
        prompt     = argument.prop.prompt

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
        show_pbar       = kwargs.get("progress", False)
        language        = kwargs.get("language", "en")
        temperature     = kwargs.get("temperature", (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
        word_timestamps = kwargs.get("word_timestamps", False)

        raw_result = []
        if prompt == 'detect_language':
            #@NOTE: the accuracy of mel spectrogram is not good enough; don't use it to transcribe
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            _, probs = self.model.detect_language(mel)
            rsp = max(probs, key=probs.get)
        elif prompt == 'decode':
            if show_pbar:
                # supress tqdm warning
                from tqdm import tqdm
                pbar = tqdm(self._get_chunks(audio))
            else:
                pbar = self._get_chunks(audio)
            for chunk in pbar:
                result = self.model.transcribe(
                    chunk,
                    language=language,
                    word_timestamps=word_timestamps,
                    temperature=temperature,
                    fp16=False,
                )
                raw_result.append(result)
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

        metadata = {}

        rsp = WhisperResult(rsp)
        if raw_result:
            rsp.raw = raw_result
        return [rsp], metadata

    def prepare(self, argument):
        assert not argument.prop.processed_input, "Whisper does not support processed_input."
        assert argument.prop.audio, "Whisper requires audio input."
        audio_file  = str(argument.prop.audio)
        audio       = whisper.load_audio(audio_file)
        argument.prop.prepared_input = (audio_file, audio)

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
