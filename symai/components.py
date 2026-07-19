import inspect
import logging
import sys
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from pydoc import locate
from random import sample
from string import ascii_lowercase, ascii_uppercase
from threading import Lock

from beartype import beartype
from tqdm import tqdm

from symai import core
from symai.backend.base import Engine
from symai.context import CURRENT_ENGINE_VAR
from symai.post_processors import PostProcessor
from symai.pre_processors import PreProcessor
from symai.prompts import Prompt
from symai.symbol import Expression, Symbol
from symai.utils import Extra, missing_dependency

logger = logging.getLogger(__name__)


class Interface(Expression):
    """Factory resolving a bundled interface by name to an instance of its class.

    ``Interface("local_search")`` imports ``symai.extended.interfaces.local_search``
    and returns an instance of its ``local_search`` class.
    """

    def __new__(cls, module: str, *args, **kwargs):
        name = str(module).lower().replace("-", "_")
        expression_cls = Interface.load_module_class(name)
        if expression_cls is None:
            msg = f"No interface named {name!r}."
            raise ValueError(msg)
        return expression_cls(*args, **kwargs)

    @staticmethod
    def load_module_class(name: str):
        module = locate(f"symai.extended.interfaces.{name}")
        if module is None:
            return None
        return getattr(module, name, None)


# @TODO: BinPacker(format="...") -> ensure that data packages form a "bin" that's consistent (e.g. never break a sentence in the middle)
class FileReader(Expression):
    @staticmethod
    def exists(path: str) -> bool:
        # remove slicing if any
        _tmp = path
        _splits = _tmp.split("[")
        if "[" in _tmp:
            _tmp = _splits[0]
        assert len(_splits) == 1 or len(_splits) == 2, "Invalid file link format."
        _tmp = Path(_tmp)
        # check if file exists and is a file
        return _tmp.is_file()

    @staticmethod
    def get_files(folder_path: str, max_depth: int = 1) -> list[str]:
        accepted_formats = [
            ".pdf",
            ".md",
            ".txt",
            ".py",
            ".json",
            ".yaml",
            ".yml",
            ".csv",
            ".tsv",
            ".log",
            ".docx",
            ".pptx",
            ".xlsx",
            ".xls",
            ".toml",
            ".html",
            ".htm",
            ".xml",
            ".epub",
            ".ipynb",
            ".zip",
            ".jpg",
            ".jpeg",
            ".png",
        ]

        folder = Path(folder_path)
        files = []
        for file_path in folder.rglob("*"):
            if file_path.is_file() and file_path.suffix in accepted_formats:
                relative_path = file_path.relative_to(folder)
                depth = len(relative_path.parts) - 1
                if depth <= max_depth:
                    files.append(file_path.as_posix())
        return files

    @staticmethod
    def integrity_check(files: list[str]) -> list[str]:
        not_skipped = []
        for file in tqdm(files):
            if FileReader.exists(file):
                not_skipped.append(file)
            else:
                logger.warning("Skipping file: %s", file)
        return not_skipped

    @staticmethod
    def _read_file(path, **kwargs):
        """Top-level helper for multiprocessing — returns a plain string."""
        from symai import Symbol  # noqa: PLC0415

        return Symbol(path).open(**kwargs).value

    def forward(self, files: str | list[str], workers: int = 1, **kwargs) -> Expression:
        assert workers >= 1, f"workers must be >= 1, got {workers}"
        if isinstance(files, str):
            files = [files]
        if kwargs.get("run_integrity_check"):
            files = self.integrity_check(files)
        if workers == 1:
            results = [self.open(f, **kwargs).value for f in files]
        else:
            results = [None] * len(files)
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._read_file, f, **kwargs): i for i, f in enumerate(files)
                }
                for fut in as_completed(futures):
                    results[futures[fut]] = fut.result()
        return self.sym_return_type(results)


class Function(Expression):
    def __init__(
        self,
        prompt: str = "",
        examples: str | None = [],
        pre_processors: list[PreProcessor] | None = None,
        post_processors: list[PostProcessor] | None = None,
        default: object | None = None,
        constraints: list[Callable] | None = None,
        return_type: type | None = str,
        sym_return_type: type | None = Symbol,
        origin_type: type | None = Expression,
        *args,
        **kwargs,
    ):
        if constraints is None:
            constraints = []
        super().__init__(**kwargs)
        chars = ascii_lowercase + ascii_uppercase
        self.name = "func_" + "".join(sample(chars, 15))
        self.args = args
        self.kwargs = kwargs
        self._promptTemplate = prompt
        self._promptFormatArgs = []
        self._promptFormatKwargs = {}
        self.examples = Prompt(examples)
        self.pre_processors = pre_processors
        self.post_processors = post_processors
        self.constraints = constraints
        self.default = default
        self.return_type = return_type
        self.sym_return_type = sym_return_type
        self.origin_type = origin_type

    @property
    def prompt(self):
        # return a copy of the prompt template
        if len(self._promptFormatArgs) == 0 and len(self._promptFormatKwargs) == 0:
            return self._promptTemplate
        return f"{self._promptTemplate}".format(*self._promptFormatArgs, **self._promptFormatKwargs)

    def format(self, *args, **kwargs):
        self._promptFormatArgs = args
        self._promptFormatKwargs = kwargs

    def forward(self, *args, **kwargs) -> Expression:
        # special case for few shot function prompt definition override
        if "fn" in kwargs:
            self.prompt = kwargs["fn"]
            del kwargs["fn"]

        @core.few_shot(
            *self.args,
            prompt=self.prompt,
            examples=self.examples,
            pre_processors=self.pre_processors,
            post_processors=self.post_processors,
            constraints=self.constraints,
            default=self.default,
            **self.kwargs,
        )
        def _func(_, *args, **kwargs) -> self.return_type:
            pass

        _type = type(
            self.name,
            (self.origin_type,),
            {
                # constructor
                "forward": _func,
                "sym_return_type": self.sym_return_type,
                "static_context": self.static_context,
                "dynamic_context": self.dynamic_context,
                "__class__": self.__class__,
                "__module__": self.__module__,
            },
        )
        obj = _type()

        return self._to_symbol(obj(*args, **kwargs))


class PrimitiveDisabler(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._primitives = set()
        self._original_primitives = defaultdict(list)

    def __enter__(self):
        # Import Symbol lazily so components does not clash with symbol during load.
        from symai.symbol import Symbol  # noqa

        frame = inspect.currentframe()
        f_locals = frame.f_back.f_locals
        self._symbols = {key: value for key, value in f_locals.items() if isinstance(value, Symbol)}
        self._extract_primitives()
        self._disable_primitives()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._enable_primitives()

    def _disable_primitives(self):
        for sym_name, sym in self._symbols.items():
            for func in self._primitives:
                if hasattr(sym, func):
                    self._original_primitives[sym_name].append((func, getattr(sym, func)))
                    setattr(sym, func, lambda *_args, **_kwargs: None)

    def _enable_primitives(self):
        for sym_name, sym in self._symbols.items():
            for func, value in self._original_primitives[sym_name]:
                setattr(sym, func, value)

    def _extract_primitives(self):
        for sym in self._symbols.values():
            for primitive in sym._primitives:
                for method, _ in inspect.getmembers(primitive, predicate=inspect.isfunction):
                    if method in self._primitives or method.startswith("_"):
                        continue
                    self._primitives.add(method)


class SelfPrompt(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, existing_prompt: dict[str, str], **kwargs) -> dict[str, str]:
        return Engine.self_prompt(existing_prompt, **kwargs)


class MetadataTracker(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trace = False
        self._original_trace = None
        self._metadata = {}
        self._engines_by_metadata_id = {}
        self._metadata_id = 0

    def __str__(self, value=None):
        value = value or self.metadata
        if isinstance(value, dict):
            return (
                "{\n\t"
                + ", \n\t".join(f'"{k}": {self.__str__(v)}' for k, v in value.items())
                + "\n}"
            )
        if isinstance(value, list):
            return "[" + ", ".join(self.__str__(item) for item in value) + "]"
        if isinstance(value, str):
            return f'"{value}"'
        return f"\n\t    {value}"

    def __new__(cls, *_args, **_kwargs):
        cls._lock = getattr(cls, "_lock", Lock())
        with cls._lock:
            instance = super().__new__(cls)
            instance._metadata = {}
            instance._engines_by_metadata_id = {}
            instance._metadata_id = 0
            return instance

    def __enter__(self):
        self._trace = True
        # Set up trace function to intercept calls
        self._original_trace = sys.gettrace()
        sys.settrace(self._trace_calls)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._trace = False
        sys.settrace(self._original_trace)

    def _trace_calls(self, frame, event, arg):
        if not self._trace:
            return None

        if (
            event == "return"
            and frame.f_code.co_name == "forward"
            and "self" in frame.f_locals
            and isinstance(frame.f_locals["self"], Engine)
            and arg is not None  # Ensure arg is not None to avoid unpacking error on exceptions
        ):
            _, metadata = arg  # arg contains return value on 'return' event
            engine_name = frame.f_locals["self"].__class__.__name__
            # getattr fallback: not all Engine subclasses set self.model (e.g. FileEngine)
            model_name = getattr(frame.f_locals["self"], "model", None)
            self._metadata[(self._metadata_id, engine_name, model_name)] = metadata
            self._engines_by_metadata_id[self._metadata_id] = frame.f_locals["self"]
            self._metadata_id += 1

        return self._trace_calls

    def _accumulate_usage_record(self, token_details, engine_name, model_name, record):
        details = token_details[(engine_name, model_name)]
        details["usage"]["prompt_tokens"] += record.prompt_tokens
        details["usage"]["completion_tokens"] += record.completion_tokens
        details["usage"]["total_tokens"] += record.total_tokens
        details["usage"]["total_calls"] += record.total_calls

        for key, value in record.prompt_breakdown.items():
            details["prompt_breakdown"][key] += value
        for key, value in record.completion_breakdown.items():
            details["completion_breakdown"][key] += value

        details["prompt_breakdown"]["cached_tokens"] += 0
        details["completion_breakdown"]["reasoning_tokens"] += 0

        if record.extras:
            extras = details.setdefault("extras", {})
            for key, value in record.extras.items():
                extras[key] = extras.get(key, 0) + value

    def _accumulate_completion_token_details(self):
        """Parses the return object and accumulates completion token details per token type"""
        if not self._metadata:
            logger.warning("No metadata available to generate usage details.")
            return {}

        token_details = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # Note on try/except:
        # The unpacking shouldn't fail; if it fails, it's likely the API response format has changed and we need to know that ASAP
        for (metadata_id, engine_name, model_name), metadata in self._metadata.items():
            try:
                engine = self._engines_by_metadata_id.get(metadata_id)
                if engine is not None:
                    usage_record = engine.usage_record_from_metadata(metadata)
                    if usage_record is not None:
                        self._accumulate_usage_record(
                            token_details,
                            engine_name,
                            model_name,
                            usage_record,
                        )
                        continue

                if engine_name == "ParallelEngine":
                    token_details[(engine_name, None)]["usage"]["total_calls"] += 1
                    # There are no model-specific tokens for this engine
                    token_details[(engine_name, None)]["usage"]["completion_tokens"] += 0
                    token_details[(engine_name, None)]["usage"]["prompt_tokens"] += 0
                    token_details[(engine_name, None)]["usage"]["total_tokens"] += 0
                    #!: Backward compatibility for components like `RuntimeInfo`
                    token_details[(engine_name, None)]["prompt_breakdown"]["cached_tokens"] += (
                        0  # Assignment not allowed with defualtdict
                    )
                    token_details[(engine_name, None)]["completion_breakdown"][
                        "reasoning_tokens"
                    ] += 0
                    self._track_parallel_usage_items(token_details, engine_name, metadata)
                elif engine_name == "EmbeddingEngine":
                    usage = metadata["raw_output"].usage
                    token_details[(engine_name, model_name)]["usage"]["prompt_tokens"] += (
                        usage.prompt_tokens
                    )
                    token_details[(engine_name, model_name)]["usage"]["completion_tokens"] += 0
                    token_details[(engine_name, model_name)]["usage"]["total_tokens"] += (
                        usage.total_tokens
                    )
                    token_details[(engine_name, model_name)]["usage"]["total_calls"] += 1
                    token_details[(engine_name, model_name)]["prompt_breakdown"][
                        "cached_tokens"
                    ] += 0
                    token_details[(engine_name, model_name)]["completion_breakdown"][
                        "reasoning_tokens"
                    ] += 0
                elif engine_name == "OpenAISearchEngine":
                    usage = metadata["raw_output"].usage
                    token_details[(engine_name, model_name)]["usage"]["prompt_tokens"] += (
                        usage.input_tokens
                    )
                    token_details[(engine_name, model_name)]["usage"]["completion_tokens"] += (
                        usage.output_tokens
                    )
                    token_details[(engine_name, model_name)]["usage"]["total_tokens"] += (
                        usage.total_tokens
                    )
                    token_details[(engine_name, model_name)]["usage"]["total_calls"] += 1
                    token_details[(engine_name, model_name)]["prompt_breakdown"][
                        "cached_tokens"
                    ] += usage.input_tokens_details.cached_tokens
                    token_details[(engine_name, model_name)]["completion_breakdown"][
                        "reasoning_tokens"
                    ] += usage.output_tokens_details.reasoning_tokens
                elif engine_name == "GeminiSearchEngine":
                    # NOTE: Gemini grounding exposes token accounting on `interaction.usage`
                    # with its own field names (total_input_tokens, total_thought_tokens,
                    # total_cached_tokens). These fields are Optional in the SDK, so guard
                    # against None with `or 0`.
                    usage = metadata["raw_output"].usage
                    token_details[(engine_name, model_name)]["usage"]["prompt_tokens"] += (
                        usage.total_input_tokens
                    )
                    token_details[(engine_name, model_name)]["usage"]["completion_tokens"] += (
                        usage.total_output_tokens
                    )
                    token_details[(engine_name, model_name)]["usage"]["total_tokens"] += (
                        usage.total_tokens
                    )
                    token_details[(engine_name, model_name)]["usage"]["total_calls"] += 1
                    token_details[(engine_name, model_name)]["prompt_breakdown"][
                        "cached_tokens"
                    ] += getattr(usage, "total_cached_tokens", 0) or 0
                    token_details[(engine_name, model_name)]["completion_breakdown"][
                        "reasoning_tokens"
                    ] += getattr(usage, "total_thought_tokens", 0) or 0
                    # NOTE: Gemini 3 bills per *search query*, not per prompt. One forward() may
                    # bundle several queries inside a single google_search_call step, or skip
                    # search entirely (grounding_tool_count is None then). total_calls counts
                    # engine invocations, so we surface the authoritative emitted-query count
                    # (usage.grounding_tool_count) separately via extras. Both the list and the
                    # inner count are Optional in the SDK, hence the `or` guards.
                    token_details[(engine_name, model_name)]["extras"]["google_search_queries"] += (
                        sum(gtc.count or 0 for gtc in usage.grounding_tool_count or [])
                    )
                elif engine_name == "MistralOCREngine":
                    # Mistral OCR uses page-based billing, not token-based
                    raw_output = metadata.get("raw_output")
                    usage_info = getattr(raw_output, "usage_info", None)
                    pages_processed = getattr(usage_info, "pages_processed", 0) or 0
                    doc_size_bytes = getattr(usage_info, "doc_size_bytes", 0) or 0
                    token_details[(engine_name, model_name)]["usage"]["prompt_tokens"] += 0
                    token_details[(engine_name, model_name)]["usage"]["completion_tokens"] += 0
                    token_details[(engine_name, model_name)]["usage"]["total_tokens"] += 0
                    token_details[(engine_name, model_name)]["usage"]["total_calls"] += 1
                    token_details[(engine_name, model_name)]["prompt_breakdown"][
                        "cached_tokens"
                    ] += 0
                    token_details[(engine_name, model_name)]["completion_breakdown"][
                        "reasoning_tokens"
                    ] += 0
                    extras = token_details[(engine_name, model_name)].setdefault("extras", {})
                    extras["pages_processed"] = extras.get("pages_processed", 0) + pages_processed
                    extras["doc_size_bytes"] = extras.get("doc_size_bytes", 0) + doc_size_bytes
                else:
                    logger.warning("Tracking %s is not supported.", engine_name)
                    continue
            except Exception as e:
                msg = f"Failed to parse metadata for {engine_name}: {e}"
                raise AttributeError(msg) from e

        # Convert to normal dict
        return {**token_details}

    def _track_parallel_usage_items(self, token_details, engine_name, metadata):
        usage_items = getattr(metadata.get("raw_output", None), "usage", None)
        if not usage_items:
            return
        if isinstance(usage_items, dict):
            usage_items = usage_items.values()
        extras = token_details[(engine_name, None)].setdefault("extras", {})
        for item in usage_items:
            name = getattr(item, "name", None)
            count = getattr(item, "count", None)
            if name in ("sku_search", "sku_extract_excerpts") and isinstance(count, (int, float)):
                extras[name] = extras.get(name, 0) + count

    @property
    def metadata(self) -> list[dict]:
        return self._metadata

    @property
    def usage(self) -> dict[str, dict]:
        return self._accumulate_completion_token_details()


class DynamicEngine(Expression):
    """Context manager for dynamically switching neurosymbolic engine models."""

    def __init__(
        self,
        model: str,
        api_key: str,
        _debug: bool = False,
        *,
        client_timeout: float | None = None,
        client_max_retries: int | None = None,
        **_kwargs,
    ):
        super().__init__()
        self.model = model
        self.api_key = api_key
        # Client-level HTTP settings (real request timeout / retries). Callers pin these
        # via e.g. components.*.dynamic_engine_name configs; forward them to the
        # neurosymbolic engine so per-component calls get real hang protection.
        self.client_timeout = client_timeout
        self.client_max_retries = client_max_retries
        self._entered = False
        self._lock = Lock()
        self.engine_instance = None
        self._ctx_token = None

    def __new__(cls, *_args, **_kwargs):
        cls._lock = getattr(cls, "_lock", Lock())
        with cls._lock:
            instance = super().__new__(cls)
            instance._metadata = {}
            instance._metadata_id = 0
            return instance

    def __enter__(self):
        self._entered = True
        self.engine_instance = self._create_engine_instance()
        # Set ContextVar and keep the token for proper reset
        try:
            self._ctx_token = CURRENT_ENGINE_VAR.set(self.engine_instance)
        except Exception:
            self._ctx_token = None
        return self.engine_instance

    def __exit__(self, exc_type, exc_value, traceback):
        self._entered = False
        # Reset ContextVar back to previous value
        try:
            if self._ctx_token is not None:
                try:
                    CURRENT_ENGINE_VAR.reset(self._ctx_token)
                except ValueError:
                    # Token belongs to a different logical Context (e.g., run in anyio worker)
                    # Fallback: clear the var in this Context to avoid leaking the engine
                    CURRENT_ENGINE_VAR.set(None)
        finally:
            self._ctx_token = None

    def _create_engine_instance(self):
        """Create an engine instance based on the model name."""
        # Deferred to avoid components <-> neurosymbolic engine circular imports.
        from symai.backend.engines.neurosymbolic import ENGINE_MAPPING  # noqa
        from symai.backend.engines.ocr import OCR_ENGINE_MAPPING  # noqa
        from symai.backend.engines.search import SEARCH_ENGINE_MAPPING  # noqa

        try:
            # Check neurosymbolic engines first
            engine_class = ENGINE_MAPPING.get(self.model)

            # Check search engines
            if engine_class is None:
                engine_class = SEARCH_ENGINE_MAPPING.get(self.model)
                if engine_class is not None:
                    return engine_class(api_key=self.api_key)

            # Check OCR engines
            if engine_class is None:
                engine_class = OCR_ENGINE_MAPPING.get(self.model)
                if engine_class is not None:
                    return engine_class(api_key=self.api_key, model=self.model)

            if engine_class is None:
                msg = (
                    f"Unsupported model '{self.model}'. Model names are provider-prefixed "
                    "(e.g. 'openai:o3', 'anthropic:claude-opus-4-6')."
                )
                raise ValueError(msg)
            # Forward client-level HTTP settings to neurosymbolic engines (all accept
            # client_timeout / client_max_retries); the search/OCR engines above do not.
            client_kwargs = {}
            if self.client_timeout is not None:
                client_kwargs["client_timeout"] = self.client_timeout
            if self.client_max_retries is not None:
                client_kwargs["client_max_retries"] = self.client_max_retries
            return engine_class(api_key=self.api_key, model=self.model, **client_kwargs)
        except Exception as e:
            msg = f"Failed to create engine for model '{self.model}': {e!s}"
            raise ValueError(msg) from e


# Chonkie chunker imports - lazy loaded
_CHONKIE_MODULES = None
_CHUNKER_MAPPING = None
_CHONKIE_AVAILABLE = None


def _lazy_import_chonkie():
    """Lazily import chonkie modules when needed."""
    global _CHONKIE_MODULES, _CHUNKER_MAPPING, _CHONKIE_AVAILABLE

    if _CHONKIE_MODULES is not None:
        return _CHONKIE_MODULES

    try:
        from chonkie import (  # noqa
            CodeChunker,
            LateChunker,
            NeuralChunker,
            RecursiveChunker,
            SemanticChunker,
            SentenceChunker,
            SlumberChunker,
            TableChunker,
            TokenChunker,
        )
        from chonkie.embeddings.base import BaseEmbeddings  # noqa
        from tokenizers import Tokenizer  # noqa

        _CHONKIE_MODULES = {
            "CodeChunker": CodeChunker,
            "LateChunker": LateChunker,
            "NeuralChunker": NeuralChunker,
            "RecursiveChunker": RecursiveChunker,
            "SemanticChunker": SemanticChunker,
            "SentenceChunker": SentenceChunker,
            "SlumberChunker": SlumberChunker,
            "TableChunker": TableChunker,
            "TokenChunker": TokenChunker,
            "BaseEmbeddings": BaseEmbeddings,
            "Tokenizer": Tokenizer,
        }
        _CHUNKER_MAPPING = {
            "TokenChunker": TokenChunker,
            "SentenceChunker": SentenceChunker,
            "RecursiveChunker": RecursiveChunker,
            "SemanticChunker": SemanticChunker,
            "CodeChunker": CodeChunker,
            "LateChunker": LateChunker,
            "NeuralChunker": NeuralChunker,
            "SlumberChunker": SlumberChunker,
            "TableChunker": TableChunker,
        }
        _CHONKIE_AVAILABLE = True
    except ImportError:
        _CHONKIE_MODULES = {}
        _CHUNKER_MAPPING = {}
        _CHONKIE_AVAILABLE = False

    return _CHONKIE_MODULES


def _get_chunker_mapping():
    """Get the chunker mapping, lazily importing chonkie if needed."""
    if _CHUNKER_MAPPING is None:
        _lazy_import_chonkie()
    return _CHUNKER_MAPPING or {}


def _is_chonkie_available():
    """Check if chonkie is available, lazily importing if needed."""
    if _CHONKIE_AVAILABLE is None:
        _lazy_import_chonkie()
    return _CHONKIE_AVAILABLE or False


@beartype
class ChonkieChunker(Expression):
    def __init__(
        self,
        tokenizer_name: str | None = "gpt2",
        embedding_model_name: str | None = "minishlab/potion-base-8M",
        **symai_kwargs,
    ):
        super().__init__(**symai_kwargs)
        self.tokenizer_name = tokenizer_name
        self.embedding_model_name = embedding_model_name
        self._tokenizer_instance = None
        self._chunker_cache: dict[tuple, object] = {}

    def forward(
        self, data: Symbol, chunker_name: str | None = "RecursiveChunker", **chunker_kwargs
    ) -> Symbol:
        if not _is_chonkie_available():
            raise missing_dependency(Extra.QDRANT, "chonkie")
        chunker = self._resolve_chunker(chunker_name, **chunker_kwargs)
        chunks = [ChonkieChunker.clean_text(chunk.text) for chunk in chunker(data.value)]
        return self._to_symbol(chunks)

    def _get_tokenizer(self):
        """Return a cached HF Tokenizer instance, loading it once on first use."""
        if self._tokenizer_instance is None:
            chonkie_modules = _lazy_import_chonkie()
            Tokenizer = chonkie_modules.get("Tokenizer")
            if Tokenizer is None:
                raise missing_dependency(Extra.QDRANT, "tokenizers")
            self._tokenizer_instance = Tokenizer.from_pretrained(self.tokenizer_name)
        return self._tokenizer_instance

    def _resolve_chunker(self, chunker_name: str, **chunker_kwargs):
        """Resolve and instantiate a chunker by name, with instance caching."""
        chunker_mapping = _get_chunker_mapping()

        if chunker_name not in chunker_mapping:
            msg = (
                f"Chunker {chunker_name} not found. Available chunkers: {list(chunker_mapping.keys())}. "
                f"See docs (https://docs.chonkie.ai/getting-started/introduction) for more info."
            )
            raise ValueError(msg)

        # Build a hashable cache key from chunker name + kwargs
        cache_key = (chunker_name, tuple(sorted(chunker_kwargs.items())))
        cached = self._chunker_cache.get(cache_key)
        if cached is not None:
            return cached

        chunker_class = chunker_mapping[chunker_name]
        chunker = None

        # Tokenizer-based chunkers (use tokenizer_name)
        if chunker_name in ["TokenChunker", "SentenceChunker", "RecursiveChunker"]:
            chunker = chunker_class(self._get_tokenizer(), **chunker_kwargs)

        # Embedding-based chunkers (use embedding_model_name)
        elif chunker_name in ["SemanticChunker", "LateChunker"]:
            chunker = chunker_class(embedding_model=self.embedding_model_name, **chunker_kwargs)

        # CodeChunker and TableChunker use tokenizer (can use string or Tokenizer object)
        elif chunker_name in ["CodeChunker", "TableChunker"] or chunker_name == "SlumberChunker":
            if "tokenizer" not in chunker_kwargs:
                chunker_kwargs["tokenizer"] = self.tokenizer_name
            chunker = chunker_class(**chunker_kwargs)

        # NeuralChunker uses model parameter (defaults provided by chonkie)
        elif chunker_name == "NeuralChunker":
            chunker = chunker_class(**chunker_kwargs)

        else:
            msg = (
                f"Chunker {chunker_name} not properly configured. "
                f"Available chunkers: {list(chunker_mapping.keys())}."
            )
            raise ValueError(msg)

        self._chunker_cache[cache_key] = chunker
        return chunker

    @staticmethod
    def clean_text(text: str) -> str:
        """Cleans text by removing problematic characters."""
        text = text.replace("\x00", "")  # Remove null bytes (\x00)
        return text.encode("utf-8", errors="ignore").decode(
            "utf-8"
        )  # Replace invalid UTF-8 sequences
