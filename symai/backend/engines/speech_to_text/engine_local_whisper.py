import contextlib
import re
from collections.abc import Iterable
from itertools import takewhile

import torch

from ....symbol import Expression, Result
from ....utils import UserMessage
from ...base import Engine
from ...settings import SYMAI_CONFIG

try:
    import whisper
    from whisper.audio import N_SAMPLES  # @NOTE: sample_rate (16_000) * chunk_length (30) = 480_000
    from whisper.tokenizer import get_tokenizer
except ImportError:
    whisper   = None
    N_SAMPLES = 16_000 * 30


class WhisperTimestampsFormatter(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, response: list[str]) -> str:
        result = []
        for i, interval in enumerate(response):
            interval_tokens = self._filter_empty_string(interval)
            prev_end = 0.0
            prev_start = 0.0
            for head, tail in zip(interval_tokens[::2], interval_tokens[1::2], strict=False):
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
                end = 30 if start + delta > 30 else start + delta
                prev_end = end
                result.append(f"{self._format_to_hours(start + (i*30))} {self._get_sentence(head)}")
        return "\n".join(result)

    def _filter_empty_string(self, s: str) -> list[str]:
        return list(filter(lambda x: x, s.split("<|")))

    def _get_timestamp(self, s: str) -> float:
        return float("".join(list(takewhile(lambda x: x != "|", s))))

    def _get_sentence(self, s: str) -> str:
        return s.split("|>")[-1]

    def _format_to_hours(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        seconds %= 3600
        minutes = int(seconds // 60)
        seconds %= 60
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}"


class WhisperResult(Result):
    def __init__(self, value, **kwargs) -> None:
        super().__init__(value, **kwargs)
        self.raw = None
        self._value = value

    def get_bins(self, bin_size_s: int = 5 * 60) -> list[str]:
        tmps = list(map(self._seconds, re.findall(r"\b\d{2}:\d{2}:\d{2}\b", self._value)))
        value_pairs = list(zip(tmps, self._value.split("\n"), strict=False))
        bin_segments = []
        result = []
        for tmp, seg in value_pairs:
            bin_segments.append(seg)
            if tmp == 0 or (tmp - bin_size_s) % bin_size_s != 0:
                continue
            result.append("\n".join(bin_segments))
            bin_segments = []
        result.append("\n".join(bin_segments))
        return result

    def _seconds(self, tmp: str) -> int:
        h, m ,s = tmp.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)


class WhisperEngine(Engine):
    def __init__(self, model: str | None = None, to_device: str | None = None):
        super().__init__()
        self.config = SYMAI_CONFIG
        self.model = None # lazy loading
        self.model_id = self.config['SPEECH_TO_TEXT_ENGINE_MODEL'] if model is None else model
        self.old_model_id = self.config['SPEECH_TO_TEXT_ENGINE_MODEL'] if model is None else model
        self.tokens = []
        self.text = []
        self.formatter = WhisperTimestampsFormatter()
        self.name = self.__class__.__name__
        if self.model is None or self.model_id != self.old_model_id:
            device_fallback = 'cpu'
            device = "cuda" if torch.cuda.is_available() else device_fallback
            device = to_device if to_device is not None else device_fallback # user preference over auto detection
            try:
                self.model = whisper.load_model(self.model_id, device=device)
            except RuntimeError:
                UserMessage(f"Whisper failed to load model on device {device}. Fallback to {device_fallback}.")
                self.model = whisper.load_model(self.model_id, device=device_fallback)
            self.old_model_id = self.model_id

        self._try_compile()

    def id(self) -> str:
        if self.config['SPEECH_TO_TEXT_ENGINE_MODEL']:
            if whisper is None:
                UserMessage("Whisper is not installed. Please install it with `pip install symbolicai[whisper]`", raise_with=ImportError)
            return 'speech-to-text'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'SPEECH_TO_TEXT_ENGINE_MODEL' in kwargs:
            self.model_id = kwargs['SPEECH_TO_TEXT_ENGINE_MODEL']

    def forward(self, argument):
        assert whisper is not None, "Whisper is not installed. Please install it first."
        kwargs = argument.kwargs
        (_, audio) = argument.prop.prepared_input
        prompt = argument.prop.prompt

        show_pbar = kwargs.get("progress", False)
        language = kwargs.get("language", "en")
        temperature = kwargs.get("temperature", (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
        without_timestamps = kwargs.get("without_timestamps", False)

        raw_result = []
        if prompt == 'detect_language':
            #@NOTE: the accuracy of mel spectrogram is not good enough; don't use it to transcribe
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            _, probs = self.model.detect_language(mel)
            rsp = max(probs, key=probs.get)
        elif prompt == 'decode':
            if show_pbar:
                # Suppress tqdm warning; keep optional dependency lazy.
                from tqdm import tqdm # noqa
                pbar = tqdm(self._get_chunks(audio))
            else:
                pbar = self._get_chunks(audio)
            for chunk in pbar:
                result = self.model.transcribe(
                    chunk,
                    language=language,
                    without_timestamps=without_timestamps,
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
            if without_timestamps is not None:
                tokenizer = get_tokenizer(self.model.is_multilingual)
                tokens = [tokenizer.decode_with_timestamps(t) for t in self.tokens]
                rsp = self.formatter(tokens)
            else:
                rsp = " ".join(self.text)
        else:
            UserMessage(f"Unknown whisper command prompt: {prompt}", raise_with=ValueError)

        metadata = {}
        rsp = WhisperResult(rsp)
        if raw_result:
            rsp.raw = raw_result
        return [rsp], metadata

    def prepare(self, argument):
        assert not argument.prop.processed_input, "Whisper does not support processed_input."
        assert argument.prop.audio, "Whisper requires audio input."
        audio_file  = str(argument.prop.audio)
        audio = whisper.load_audio(audio_file)
        argument.prop.prepared_input = (audio_file, audio)

    def _get_chunks(self, it: Iterable, batch: int = N_SAMPLES) -> Iterable[torch.Tensor]:
        """
        Split an iterable into chunks of size `batch`. It defaults to `N_SAMPLES` 480_000 samples which is equal to 30 seconds.
        """
        size = len(it)
        for i in range(0, size, batch):
            yield torch.tensor(it[i:min(i + batch, size)]).to(self.model.device)

    def _try_compile(self):
        with contextlib.suppress(Exception):
            self.model = torch.compile(self.model)
