import pytest
from pydantic import SecretStr, ValidationError

from symai.providers._client.models import StrictModel
from symai.providers._client.settings import HttpProviderSettings
from symai.providers.cerebras.settings import ChatCompletionsSettings as CerebrasSettings
from symai.providers.deepseek.settings import ChatCompletionsSettings as DeepSeekSettings
from symai.providers.openai.settings import EmbeddingSettings, ResponsesSettings


@pytest.mark.parametrize(
    "settings_type",
    [CerebrasSettings, DeepSeekSettings, EmbeddingSettings, ResponsesSettings],
)
def test_provider_settings_use_runtime_blind_shared_http_settings(settings_type):
    assert issubclass(settings_type, HttpProviderSettings)
    assert issubclass(settings_type, StrictModel)


def test_http_provider_settings_preserve_defaults_and_strict_validation():
    settings = HttpProviderSettings(api_key=SecretStr("key"), model="model")

    assert settings.request_timeout == 600.0
    assert settings.connect_timeout == 10.0
    assert settings.connect_retries == 0
    with pytest.raises(ValidationError):
        HttpProviderSettings.model_validate(
            {"api_key": "key", "model": "model", "connect_retries": "1"}
        )
