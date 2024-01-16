from ...base import Engine

from ...settings import SYMAI_CONFIG


class PluginEmbeddingEngine(Engine):
    def id(self) -> str:
        if not SYMAI_CONFIG['EMBEDDING_ENGINE_API_KEY'] or SYMAI_CONFIG['EMBEDDING_ENGINE_API_KEY'] == '':
            from ....functional import EngineRepository
            # Register the embedding engine from the plugin
            EngineRepository.register_from_plugin('embedding', plugin='ExtensityAI/embeddings', kwargs={'model': 'all-mpnet-base-v2'}, allow_engine_override=True)
        return super().id() # do not register this engine as we want the plugin to be used
