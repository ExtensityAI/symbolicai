import importlib

import pytest

mandatory_dependencies = [
    "attrs",
    "setuptools",
    "toml",
    "natsort",
    "numpy",
    "tqdm",
    "box",
    "pandas",
    "sklearn",
    "torch",
    "torchaudio",
    "torchvision",
    "yaml",
    "transformers",
    "sympy",
    "openai",
    "anthropic",
    "pypdf",
    "IPython",
    "accelerate",
    "sentencepiece",
    "sentence_transformers",
    "tiktoken",
    "tika",
    "bs4",
    "colorama",
    "git",
    "pathos",
    "prompt_toolkit",
    "pydub",
    "cv2",
    "pymongo",
    "requests_toolbelt",
    "pyvis",
    "beartype",
    "pydantic",
    "pydantic_core",
    "pydantic_settings",
    "Crypto",
]

optional_dependencies = {
    "bitsandbytes": ["bitsandbytes"],
    "blip2": ["decord", "lavis", "cv2"],
    "hf": ["transformers", "accelerate", "peft", "datasets", "trl"],
    "wolframalpha": ["wolframalpha"],
    "whisper": ["whisper"],
    "selenium": ["selenium", "webdriver_manager", "chromedriver_autoinstaller"],
    "serpapi": ["serpapi"],
    "pinecone": ["pinecone"],
    "bard": ["bardapi"],
    "services": ["fastapi", "redis", "uvicorn"],
    "solver": ["z3"],
}

@pytest.mark.parametrize("module_name", mandatory_dependencies)
def test_mandatory_dependencies(module_name):
    importlib.import_module(module_name)

@pytest.mark.parametrize("group, modules", optional_dependencies.items())
def test_optional_dependencies(group, modules):
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except pytest.fail.Exception:
            assert False, f"Optional dependency {module_name} from group {group} is not installed"

def test_symai():
    try:
        importlib.import_module("symai")
    except pytest.fail.Exception:
        assert False, "Failed to import symai"

