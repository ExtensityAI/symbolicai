from .engine_mistral import MistralOCREngine

OCR_ENGINE_MAPPING = {
    "mistral": MistralOCREngine,
}

__all__ = [
    "OCR_ENGINE_MAPPING",
    "MistralOCREngine",
]
