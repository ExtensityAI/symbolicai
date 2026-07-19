import pytest

from symai.backend.engines.text_to_speech.openai import TTSEngine
from symai.backend.engines.text_to_speech.openai.models import (
    API_PINNED,
    OPENAI_SPEECH_CONTENT_TYPES,
    OPENAI_SPEECH_URL,
)
from tests.engines.text_to_speech.interface import (
    MOCK_PROMPT,
    MOCK_VOICE,
    TextToSpeechTestInterface,
)


class TestOpenAITTSEngine(TextToSpeechTestInterface):
    engine_cls = TTSEngine
    default_model = "tts-1"
    wire_url = OPENAI_SPEECH_URL
    auth_header_name = "Authorization"
    auth_header_prefix = "Bearer "
    api_pinned = API_PINNED
    api_pinned_module = "symai.backend.engines.text_to_speech.openai.models"
    api_key_env = "OPENAI_API_KEY"
    mock_content_type = OPENAI_SPEECH_CONTENT_TYPES["mp3"]
    live_model = "tts-1"

    def expected_request_body_subset(self):
        return {"model": self.default_model, "input": MOCK_PROMPT, "voice": MOCK_VOICE.lower()}

    def test_default_wire_body_omits_optional_fields(self, tmp_path):
        api, _output, _metadata = self.forward_through_mock(tmp_path / "speech.mp3")

        # exclude_none: response_format/speed stay off the wire unless requested
        assert set(api.last_body) == {"model", "input", "voice"}

    @pytest.mark.engine_live
    def test_live_wav_format(self, engine_api_mode, tmp_path):
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(api_key)
        audio_path = tmp_path / "speech.wav"
        argument = self.make_argument(
            audio_path,
            prompt="Hello from the symbolicai engine test.",
            kwargs={"response_format": "wav"},
        )
        engine.prepare(argument)
        output, metadata = engine.forward(argument)

        audio = output[0].value
        assert isinstance(audio, bytes) and len(audio) > 100
        assert metadata["content_type"] == OPENAI_SPEECH_CONTENT_TYPES["wav"]
        assert audio[:4] == b"RIFF" and audio[8:12] == b"WAVE"
        assert audio_path.read_bytes() == audio
