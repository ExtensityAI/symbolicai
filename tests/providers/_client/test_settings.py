import pytest
from pydantic import SecretStr, ValidationError

from symai.providers._client.models import StrictModel
from symai.providers._client.settings import HttpProviderSettings


def test_http_provider_settings_are_a_runtime_blind_strict_model():
    assert issubclass(HttpProviderSettings, StrictModel)


def test_http_provider_settings_preserve_defaults_and_strict_validation():
    settings = HttpProviderSettings(api_key=SecretStr("key"), model="model")

    assert settings.request_timeout == 600.0
    assert settings.connect_timeout == 10.0
    assert settings.connect_retries == 0
    with pytest.raises(ValidationError):
        HttpProviderSettings.model_validate(
            {"api_key": "key", "model": "model", "connect_retries": "1"}
        )


def test_http_provider_settings_reject_unknown_fields():
    payload = {"api_key": SecretStr("key"), "model": "model"}

    assert HttpProviderSettings.model_validate(payload).model == "model"
    with pytest.raises(ValidationError):
        HttpProviderSettings.model_validate({**payload, "model_path": "/models/local.gguf"})
