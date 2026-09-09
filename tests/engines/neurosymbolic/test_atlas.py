from symai.backend.engines.neurosymbolic.atlas import (
    API_PINNED,
    ATLAS_CHAT_COMPLETIONS_URL,
    ATLAS_MODEL_SPECS,
    SUPPORTED_ATLAS_MODELS,
    AtlasEngine,
    atlas_model_spec_for,
    atlas_strip_prefix,
)
from symai.backend.engines.neurosymbolic.deepseek.models import DeepSeekResponse
from tests.engines.neurosymbolic.test_deepseek import (
    TestDeepSeekEngine as _TestDeepSeekEngine,
)


class TestAtlasEngine(_TestDeepSeekEngine):
    engine_cls = AtlasEngine
    supported_models = tuple(SUPPORTED_ATLAS_MODELS)
    model_specs = ATLAS_MODEL_SPECS
    default_model = "atlas:deepseek-ai/deepseek-v4-flash"
    response_cls = DeepSeekResponse
    wire_provider = "atlas"
    wire_url = ATLAS_CHAT_COMPLETIONS_URL
    api_pinned = API_PINNED

    def spec_for(self, model):
        return atlas_model_spec_for(model)

    def expected_wire_model(self, model=None):
        return atlas_strip_prefix(model or self.default_model)

    def test_build_request_strips_provider_prefix_from_wire_model(self):
        request = self.make_engine().build_request(self.make_prepared_argument())

        assert request.body()["model"] == self.expected_wire_model()
