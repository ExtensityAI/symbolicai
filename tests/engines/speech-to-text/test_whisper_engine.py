from pathlib import Path

import pytest

from symai.backend.engines.speech_to_text.engine_local_whisper import WhisperResult
from symai.backend.settings import SYMAI_CONFIG
from symai.extended import Interface
from symai.utils import semassert

try:
    import whisper  # noqa: F401
except ImportError as e:
    msg = "whisper is not installed. Please install it."
    raise ImportError(msg) from e

if SYMAI_CONFIG.get("SPEECH_TO_TEXT_ENGINE_MODEL") not in [
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "turbo",
]:
    msg = "The model you have selected is not supported by the whisper engine. Please select a supported model: [tiny, base, small, medium, large, turbo]"
    raise ValueError(msg)

model = SYMAI_CONFIG.get("SPEECH_TO_TEXT_ENGINE_MODEL")
audiofile = (Path(__file__).parent.parent.parent / "data/sample.mp3").as_posix()


def test_whisper_transcribe():
    stt = Interface("whisper")
    rsp = stt(audiofile)
    assert isinstance(rsp, WhisperResult), f"Expected WhisperResult, got {type(rsp)}"
    semassert("may have" in rsp)
    bins = rsp.get_bins()
    assert isinstance(bins, list), f"Expected list, got {type(bins)}"
    assert len(bins) == 1, f"Expected 1 bin, got {len(bins)}"


@pytest.mark.skipif(model == "turbo", reason="Turbo model is not supported for language detection")
def test_whisper_language_detection():
    stt = Interface("whisper")
    rsp = stt(audiofile, "detect_language")
    assert isinstance(rsp, WhisperResult), f"Expected WhisperResult, got {type(rsp)}"
    semassert(rsp == "en")
