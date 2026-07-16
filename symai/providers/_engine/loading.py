from collections.abc import Callable, Mapping

from symai.providers._client.settings import HttpProviderSettings
from symai.runtime.errors import UnsupportedModelError


def resolve_http_engine[EngineT](
    settings: Mapping[str, object],
    *,
    model_specs: Mapping[str, object],
    unsupported_model_message: str,
    client: Callable[..., object],
    engine: Callable[..., EngineT],
) -> Callable[[], EngineT]:
    """Validate settings now, and allocate transport only when the returned factory runs.

    Loading is two-phase (FIXPLAN §2) so the runtime can reject any misconfigured engine
    before a single HTTP client exists: a typo in the last engine must not leave the first
    engine's transport allocated and then torn down. Every HTTP provider needs exactly
    this shape, and the ordering is the whole invariant — validating inside `construct`
    instead would silently reintroduce the leak. Sharing it means a new provider states
    its schema and policy delta rather than re-deriving the rule.

    Raises:
        UnsupportedModelError: if `settings["model"]` is absent from `model_specs`.
        pydantic.ValidationError: if the settings are not valid `HttpProviderSettings`.
    """
    parsed = HttpProviderSettings.model_validate(dict(settings))
    if parsed.model not in model_specs:
        msg = unsupported_model_message.format(model=parsed.model)
        raise UnsupportedModelError(msg)

    def construct() -> EngineT:
        import httpx

        return engine(
            client=client(
                api_key=parsed.api_key,
                timeout=httpx.Timeout(
                    parsed.request_timeout,
                    connect=parsed.connect_timeout,
                ),
                connect_retries=parsed.connect_retries,
            ),
            model=parsed.model,
        )

    return construct
