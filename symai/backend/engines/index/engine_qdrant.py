import itertools
import logging
import tempfile
import urllib.request
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from .... import core_ext
from ....symbol import Result, Symbol
from ....utils import UserMessage
from ...base import Engine
from ...settings import SYMAI_CONFIG, SYMSERVER_CONFIG

warnings.filterwarnings("ignore", module="qdrant_client")
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import (
        Distance,
        Filter,
        NamedVector,
        PointStruct,
        Query,
        ScoredPoint,
        VectorParams,
    )
except ImportError:
    QdrantClient = None
    models = None
    Distance = None
    VectorParams = None
    PointStruct = None
    Filter = None
    Query = None
    NamedVector = None
    ScoredPoint = None

try:
    from ....components import ChonkieChunker, FileReader
except ImportError:
    ChonkieChunker = None
    FileReader = None

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None

logging.getLogger("qdrant_client").setLevel(logging.ERROR)


def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = list(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = list(itertools.islice(it, batch_size))


class QdrantResult(Result):
    def __init__(self, res, query: str, embedding: list, **kwargs):
        super().__init__(res, **kwargs)
        self.raw = res
        self._query = query
        self._value = self._process(res)
        self._metadata.raw = embedding

    def _process(self, res):
        if not res:
            return None
        try:
            # Qdrant returns a list of ScoredPoint objects
            # Convert to format similar to Pinecone for consistency
            if isinstance(res, list):
                matches = []
                for point in res:
                    match = {
                        "id": point.id if hasattr(point, "id") else None,
                        "score": point.score if hasattr(point, "score") else None,
                        "metadata": point.payload if hasattr(point, "payload") else {},
                    }
                    # Extract text from payload if available
                    if "text" in match["metadata"]:
                        match["metadata"]["text"] = match["metadata"]["text"]
                    elif "content" in match["metadata"]:
                        match["metadata"]["text"] = match["metadata"]["content"]
                    matches.append(match)
                return [v["metadata"].get("text", str(v)) for v in matches if "metadata" in v]
            res = self._to_symbol(res).ast()
            return [v["metadata"]["text"] for v in res.get("matches", []) if "metadata" in v]
        except Exception as e:
            message = [
                "Sorry, failed to interact with Qdrant index. Please check collection name and try again later:",
                str(e),
            ]
            return [{"metadata": {"text": "\n".join(message)}}]

    def _unpack_matches(self):
        if not self.value:
            return

        for i, match_item in enumerate(self.value):
            if isinstance(match_item, dict):
                match_text = match_item.get("metadata", {}).get("text", str(match_item))
            else:
                match_text = str(match_item)
            match_text = match_text.strip()
            if match_text.startswith("# ----[FILE_START]") and "# ----[FILE_END]" in match_text:
                m = match_text.split("[FILE_CONTENT]:")[-1].strip()
                splits = m.split("# ----[FILE_END]")
                assert len(splits) >= 2, f"Invalid file format: {splits}"
                content_text = splits[0]
                file_name = ",".join(splits[1:])  # TODO: check why there are multiple file names
                yield file_name.strip(), content_text.strip()
            else:
                yield i + 1, match_text

    def __str__(self):
        str_view = ""
        for filename, content_text in self._unpack_matches():
            # indent each line of the content
            indented_content = "\n".join(["  " + line for line in content_text.split("\n")])
            str_view += f"* {filename}\n{indented_content}\n\n"
        return f"""
[RESULT]
{"-=-" * 13}

Query: {self._query}

{"-=-" * 13}

Matches:

{str_view}
{"-=-" * 13}
"""

    def _repr_html_(self) -> str:
        # return a nicely styled HTML list results based on retrieved documents
        doc_str = ""
        for filename, content in self._unpack_matches():
            doc_str += f'<li><a href="{filename}"><b>{filename}</a></b><br>{content}</li>\n'
        return f"<ul>{doc_str}</ul>"


@dataclass
class Citation:
    id: int
    title: str
    url: str
    start: int
    end: int

    def __hash__(self):
        return hash((self.url,))


class SearchResult(Result):
    def __init__(self, value: dict[str, Any] | Any, **kwargs) -> None:
        super().__init__(value, **kwargs)
        if isinstance(value, dict) and value.get("error"):
            UserMessage(value["error"], raise_with=ValueError)
        results = self._coerce_results(value)
        text, citations = self._build_text_and_citations(results)
        self._value = text
        self._citations = citations

    def _coerce_results(self, raw: Any) -> list[dict[str, Any]]:
        if raw is None:
            return []
        results = raw.get("results", []) if isinstance(raw, dict) else getattr(raw, "results", None)
        if not results:
            return []
        return [item for item in results if isinstance(item, dict)]

    def _source_identifier(self, item: dict[str, Any], url: str) -> str:
        for key in ("source_id", "sourceId", "sourceID", "id"):
            raw = item.get(key)
            if raw is None:
                continue
            text = str(raw).strip()
            if text:
                return text
        path = Path(urlparse(url).path)
        return path.name or path.as_posix() or url

    def _build_text_and_citations(self, results: list[dict[str, Any]]):
        pieces = []
        citations = []
        cursor = 0
        cid = 1
        separator = "\n\n---\n\n"

        for item in results:
            url = str(item.get("url") or "")
            if not url:
                continue

            title = str(item.get("title") or "")
            if not title:
                path = Path(urlparse(url).path)
                title = path.name or url

            excerpts = item.get("excerpts") or []
            excerpt_parts = [ex.strip() for ex in excerpts if isinstance(ex, str) and ex.strip()]
            if not excerpt_parts:
                continue

            combined_excerpt = "\n\n".join(excerpt_parts)
            source_id = self._source_identifier(item, url)
            block_body = combined_excerpt if not source_id else f"{source_id}\n\n{combined_excerpt}"

            if pieces:
                pieces.append(separator)
                cursor += len(separator)

            opening_tag = "<source>\n"
            pieces.append(opening_tag)
            cursor += len(opening_tag)

            pieces.append(block_body)
            cursor += len(block_body)

            closing_tag = "\n</source>"
            pieces.append(closing_tag)
            cursor += len(closing_tag)

            marker = f"[{cid}]"
            start = cursor
            pieces.append(marker)
            cursor += len(marker)

            citations.append(Citation(id=cid, title=title or url, url=url, start=start, end=cursor))
            cid += 1

        return "".join(pieces), citations

    def __str__(self) -> str:
        return str(self._value or "")

    def _repr_html_(self) -> str:
        return f"<pre>{self._value or ''}</pre>"

    def get_citations(self) -> list[Citation]:
        return self._citations


class QdrantIndexEngine(Engine):
    _default_url = "http://localhost:6333"
    _default_api_key = SYMAI_CONFIG.get("INDEXING_ENGINE_API_KEY", None)
    _default_index_name = "dataindex"
    _default_index_dims = 1536
    _default_index_top_k = 5
    _default_index_metric = "Cosine"
    _default_index_values = True
    _default_index_metadata = True
    _default_retry_tries = 20
    _default_retry_delay = 0.5
    _default_retry_max_delay = -1
    _default_retry_backoff = 1
    _default_retry_jitter = 0

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = _default_api_key,
        index_name: str = _default_index_name,
        index_dims: int = _default_index_dims,
        index_top_k: int = _default_index_top_k,
        index_metric: str = _default_index_metric,
        index_values: bool = _default_index_values,
        index_metadata: bool = _default_index_metadata,
        tries: int = _default_retry_tries,
        delay: float = _default_retry_delay,
        max_delay: int = _default_retry_max_delay,
        backoff: int = _default_retry_backoff,
        jitter: int = _default_retry_jitter,
        chunker_name: str | None = "RecursiveChunker",
        tokenizer_name: str | None = "gpt2",
        embedding_model_name: str | None = "minishlab/potion-base-8M",
    ):
        super().__init__()
        self.index_name = index_name
        self.index_dims = index_dims
        self.index_top_k = index_top_k
        self.index_values = index_values
        self.index_metadata = index_metadata
        self.index_metric = self._parse_metric(index_metric)
        # Get URL from SYMSERVER_CONFIG if available, otherwise use provided or default
        if url:
            self.url = url
        elif SYMSERVER_CONFIG.get("url"):
            self.url = SYMSERVER_CONFIG.get("url")
        elif (
            SYMSERVER_CONFIG.get("online")
            and SYMSERVER_CONFIG.get("--host")
            and SYMSERVER_CONFIG.get("--port")
        ):
            self.url = f"http://{SYMSERVER_CONFIG.get('--host')}:{SYMSERVER_CONFIG.get('--port')}"
        else:
            self.url = self._default_url
        self.api_key = api_key
        self.tries = tries
        self.delay = delay
        self.max_delay = max_delay
        self.backoff = backoff
        self.jitter = jitter
        self.client = None
        self.name = self.__class__.__name__

        # Initialize chunker and reader for manager functionality
        self.chunker_name = chunker_name
        self.tokenizer_name = tokenizer_name
        self.embedding_model_name = embedding_model_name
        self.chunker = None
        self.reader = None
        self.tokenizer = None

        # Initialize chunker if available
        if ChonkieChunker:
            try:
                self.chunker = ChonkieChunker(
                    tokenizer_name=tokenizer_name, embedding_model_name=embedding_model_name
                )
                if Tokenizer:
                    self.tokenizer = Tokenizer.from_pretrained(tokenizer_name)
            except Exception as e:
                warnings.warn(f"Failed to initialize chunker: {e}")

        # Initialize FileReader
        if FileReader:
            try:
                self.reader = FileReader()
            except Exception as e:
                warnings.warn(f"Failed to initialize FileReader: {e}")

    def _parse_metric(self, metric: str) -> Distance:
        """Convert string metric to Qdrant Distance enum."""
        if QdrantClient is None:
            return metric
        metric_map = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclidean": Distance.EUCLID,
        }
        metric_lower = metric.lower()
        return metric_map.get(metric_lower, Distance.COSINE)

    def id(self) -> str:
        # Check if Qdrant is configured (either via server or direct connection)
        if SYMSERVER_CONFIG.get("online") or self.url:
            if QdrantClient is None:
                UserMessage(
                    "Qdrant client is not installed. Please install it with `pip install qdrant-client`.",
                    raise_with=ImportError,
                )
            return "index"
        return super().id()  # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "INDEXING_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["INDEXING_ENGINE_API_KEY"]
        if "INDEXING_ENGINE_URL" in kwargs:
            self.url = kwargs["INDEXING_ENGINE_URL"]

    def _init_client(self):
        """Initialize Qdrant client if not already initialized."""
        if self.client is None:
            if QdrantClient is None:
                UserMessage(
                    "Qdrant client is not installed. Please install it with `pip install qdrant-client`.",
                    raise_with=ImportError,
                )

            client_kwargs = {"url": self.url}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key

            self.client = QdrantClient(**client_kwargs)

    def _create_collection_sync(
        self,
        collection_name: str,
        vector_size: int | None = None,
        distance: (str | Distance) | None = None,
        **kwargs,
    ):
        """Synchronous collection creation for internal use."""
        self._init_client()

        vector_size = vector_size or self.index_dims
        if isinstance(distance, str):
            distance = self._parse_metric(distance)
        else:
            distance = distance or self.index_metric

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
                **kwargs,
            )

    def _init_collection(
        self, collection_name: str, collection_dims: int, collection_metric: Distance
    ):
        """Initialize or create Qdrant collection (legacy method, uses _create_collection_sync)."""
        self._create_collection_sync(collection_name, collection_dims, collection_metric)

    def _configure_collection(self, **kwargs):
        collection_name = kwargs.get("index_name", self.index_name)
        del_ = kwargs.get("index_del", False)

        if self.client is not None and del_:
            try:
                self.client.delete_collection(collection_name=collection_name)
            except Exception as e:
                warnings.warn(f"Failed to delete collection {collection_name}: {e}")

        get_ = kwargs.get("index_get", False)
        if get_:
            # Reinitialize client to refresh collection list
            self._init_client()

    def _build_query_filter(self, raw_filter: Any) -> Filter | None:
        """Normalize various filter representations into a Qdrant Filter.

        Supports:
        - None: returns None
        - Existing Filter instance: returned as-is
        - Dict[str, Any]: converted to equality-based Filter over payload keys

        The dict form is intentionally simple and maps directly to `payload.<key>`
        equality conditions, which covers the majority of RAG use cases while
        remaining easy to serialize and pass through higher-level APIs.
        """
        if raw_filter is None or Filter is None:
            return None

        # Already a Filter instance → use directly
        if isinstance(raw_filter, Filter):
            return raw_filter

        # Simple dict → build equality-based must filter
        if isinstance(raw_filter, dict):
            if models is None:
                UserMessage(
                    "Qdrant filter models are not available. "
                    "Please install `qdrant-client` to use filtering.",
                    raise_with=ImportError,
                )

            conditions = []
            for key, value in raw_filter.items():
                # We keep semantics simple and robust: every entry is treated as an
                # equality condition on the payload key (logical AND across keys).
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )

            return Filter(must=conditions) if conditions else None

        # Fallback: pass through other representations (e.g. already-built Filter-like)
        return raw_filter

    def _prepare_points_for_upsert(
        self,
        embeddings: list | np.ndarray | Any,
        ids: list[int] | None = None,
        payloads: list[dict] | None = None,
    ) -> list[PointStruct]:
        """Prepare points for upsert from embeddings, ids, and payloads."""
        points = []

        # Normalize to list
        if isinstance(embeddings, np.ndarray):
            embeddings = [embeddings] if embeddings.ndim == 1 else list(embeddings)
        elif not isinstance(embeddings, list):
            embeddings = [embeddings]

        for i, vec in enumerate(embeddings):
            point_id = self._normalize_point_id(ids[i]) if ids and i < len(ids) else i
            payload = payloads[i] if payloads and i < len(payloads) else {}
            points.append(
                PointStruct(id=point_id, vector=self._normalize_vector(vec), payload=payload)
            )

        return points

    def forward(self, argument):
        kwargs = argument.kwargs
        embedding = argument.prop.prepared_input
        if embedding is None:
            embedding = getattr(argument.prop, "prompt", None)
        if embedding is None:
            msg = (
                "Qdrant forward() requires an embedding vector. "
                "Provide it via prepared_input or prompt before calling forward()."
            )
            raise ValueError(msg)
        query = argument.prop.ori_query
        operation = argument.prop.operation
        collection_name = argument.prop.index_name if argument.prop.index_name else self.index_name
        collection_dims = argument.prop.index_dims if argument.prop.index_dims else self.index_dims
        rsp = None

        # Initialize client
        self._init_client()

        if collection_name != self.index_name:
            assert collection_name, "Please set a valid collection name for Qdrant indexing engine."
            # switch collection
            self.index_name = collection_name
            kwargs["index_get"] = True
            self._configure_collection(**kwargs)

        treat_as_search_engine = False
        if operation == "search":
            # Ensure collection exists - fail fast if it doesn't
            self._ensure_collection_exists(collection_name)
            search_kwargs = dict(kwargs)
            index_top_k = search_kwargs.pop("index_top_k", self.index_top_k)
            # Optional search parameters
            score_threshold = search_kwargs.pop("score_threshold", None)
            # Accept both `query_filter` and `filter` for convenience
            raw_filter = search_kwargs.pop("query_filter", search_kwargs.pop("filter", None))
            query_filter = self._build_query_filter(raw_filter)
            treat_as_search_engine = bool(search_kwargs.pop("treat_as_search_engine", False))

            # Use shared search helper that already handles retries and normalization
            rsp = self._search_sync(
                collection_name=collection_name,
                query_vector=embedding,
                limit=index_top_k,
                score_threshold=score_threshold,
                query_filter=query_filter,
                **search_kwargs,
            )
        elif operation == "add":
            # Create collection if it doesn't exist (only for write operations)
            self._create_collection_sync(collection_name, collection_dims, self.index_metric)
            # Use shared point preparation method
            ids = kwargs.get("ids", None)
            payloads = kwargs.get("payloads", None)
            points = self._prepare_points_for_upsert(embedding, ids, payloads)

            # Use existing _upsert method in batches
            for points_chunk in chunks(points, batch_size=100):
                self._upsert(collection_name, points_chunk)
            rsp = None
        elif operation == "config":
            # Ensure collection exists - fail fast if it doesn't
            self._ensure_collection_exists(collection_name)
            self._configure_collection(**kwargs)
            rsp = None
        else:
            msg = "Invalid operation. Supported operations: search, add, config"
            raise ValueError(msg)

        metadata = {}

        if operation == "search" and treat_as_search_engine:
            rsp = self._format_search_results(rsp, collection_name)
        else:
            rsp = QdrantResult(rsp, query, embedding)
        return [rsp], metadata

    def prepare(self, argument):
        assert not argument.prop.processed_input, (
            "Qdrant indexing engine does not support processed_input."
        )
        argument.prop.prepared_input = argument.prop.prompt

    def _upsert(self, collection_name: str, points: list[PointStruct]):
        @core_ext.retry(
            tries=self.tries,
            delay=self.delay,
            max_delay=self.max_delay,
            backoff=self.backoff,
            jitter=self.jitter,
        )
        def _func():
            return self.client.upsert(collection_name=collection_name, points=points)

        return _func()

    def _normalize_vector(self, vector: list[float] | np.ndarray) -> list[float]:
        """Normalize vector to flat list format, handling 2D arrays and nested lists."""
        if isinstance(vector, np.ndarray):
            # Flatten if 2D (e.g., shape (1, 1536) -> (1536,))
            if vector.ndim > 1:
                vector = vector.flatten()
            return vector.tolist()
        if not isinstance(vector, list):
            vector = list(vector)

        # Handle nested lists that might have slipped through
        if vector and len(vector) > 0 and isinstance(vector[0], list):
            # Flatten nested list (e.g., [[1, 2, 3]] -> [1, 2, 3])
            if len(vector) == 1:
                vector = vector[0]
            else:
                vector = [item for sublist in vector for item in sublist]

        return vector

    def _query(self, collection_name: str, query_vector: list[float], top_k: int, **kwargs):
        @core_ext.retry(
            tries=self.tries,
            delay=self.delay,
            max_delay=self.max_delay,
            backoff=self.backoff,
            jitter=self.jitter,
        )
        def _func():
            qdrant_kwargs = dict(kwargs)
            query_vector_normalized = self._normalize_vector(query_vector)
            with_payload = qdrant_kwargs.pop("with_payload", True)
            with_vectors = qdrant_kwargs.pop("with_vectors", self.index_values)
            # qdrant-client `query_points` is strict about extra kwargs and will assert if any
            # unknown arguments are provided. Because our engine `forward()` passes decorator
            # kwargs through the stack, we must drop engine-internal fields here.
            #
            # Keep only kwargs that `qdrant_client.QdrantClient.query_points` accepts (besides
            # those we pass explicitly).
            if "filter" in qdrant_kwargs and "query_filter" not in qdrant_kwargs:
                # Convenience alias supported by our public API
                qdrant_kwargs["query_filter"] = qdrant_kwargs.pop("filter")

            allowed_qdrant_kwargs = {
                "using",
                "prefetch",
                "query_filter",
                "search_params",
                "offset",
                "score_threshold",
                "lookup_from",
                "consistency",
                "shard_key_selector",
                "timeout",
            }
            qdrant_kwargs = {k: v for k, v in qdrant_kwargs.items() if k in allowed_qdrant_kwargs}
            # For single vector collections, pass vector directly to query parameter
            # For named vector collections, use Query(near_vector=NamedVector(name="vector_name", vector=...))
            # query_points API uses query_filter (not filter) for filtering
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector_normalized,
                limit=top_k,
                with_payload=with_payload,
                with_vectors=with_vectors,
                **qdrant_kwargs,
            )
            # query_points returns QueryResponse with .points attribute, extract it
            return response.points

        return _func()

    # ==================== Manager Methods ====================

    def _check_initialization(self):
        """Check if engine is properly initialized."""
        if self.client is None:
            self._init_client()
        if self.client is None:
            msg = "Qdrant client not properly initialized."
            raise RuntimeError(msg)

    def _ensure_collection_exists(self, collection_name: str):
        """Ensure collection exists, raise error if not."""
        self._check_initialization()
        if not self.client.collection_exists(collection_name):
            msg = f"Collection '{collection_name}' does not exist"
            raise ValueError(msg)

    # ==================== Collection Management ====================

    async def create_collection(
        self,
        collection_name: str,
        vector_size: int | None = None,
        distance: (str | Distance) | None = None,
        **kwargs,
    ):
        """
        Create a new collection in Qdrant.

        Args:
            collection_name: Name of the collection to create
            vector_size: Size of the vectors in this collection (defaults to index_dims)
            distance: Distance metric (COSINE, EUCLIDEAN, or DOT) or string
            **kwargs: Additional collection configuration parameters
        """
        self._check_initialization()

        if self.client.collection_exists(collection_name):
            warnings.warn(f"Collection '{collection_name}' already exists")
            return

        # Use shared synchronous method
        self._create_collection_sync(collection_name, vector_size, distance, **kwargs)

    async def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Name of the collection to check

        Returns:
            True if collection exists, False otherwise
        """
        self._check_initialization()
        return self.client.collection_exists(collection_name)

    async def list_collections(self) -> list[str]:
        """
        List all collections in Qdrant.

        Returns:
            List of collection names
        """
        self._check_initialization()
        collections = self.client.get_collections().collections
        return [collection.name for collection in collections]

    async def delete_collection(self, collection_name: str):
        """
        Delete a collection from Qdrant.

        Args:
            collection_name: Name of the collection to delete
        """
        self._ensure_collection_exists(collection_name)
        self.client.delete_collection(collection_name)

    async def get_collection_info(self, collection_name: str) -> dict:
        """
        Get information about a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary containing collection information
        """
        self._ensure_collection_exists(collection_name)
        collection_info = self.client.get_collection(collection_name)
        # Extract vector config - handle both single vector and named vectors
        vector_config = collection_info.config.params.vectors
        if hasattr(vector_config, "size"):
            # Single vector configuration
            vectors_info = {
                "size": vector_config.size,
                "distance": vector_config.distance,
            }
        else:
            # Named vectors configuration
            vectors_info = {
                "named_vectors": {
                    name: {"size": vec.size, "distance": vec.distance}
                    for name, vec in vector_config.items()
                }
            }
        # Qdrant 1.16.1+ compatibility: vectors_count and indexed_vectors_count may not exist
        # Use points_count as the primary count, and try to get vectors_count if available
        result = {
            "name": collection_name,
            "points_count": collection_info.points_count,
            "config": {"params": {"vectors": vectors_info}},
        }

        # Try to get vectors_count if available (for older Qdrant versions)
        if hasattr(collection_info, "vectors_count"):
            result["vectors_count"] = collection_info.vectors_count
        else:
            # In Qdrant 1.16.1+, vectors_count is not available, use points_count as approximation
            result["vectors_count"] = collection_info.points_count

        # Try to get indexed_vectors_count if available
        if hasattr(collection_info, "indexed_vectors_count"):
            result["indexed_vectors_count"] = collection_info.indexed_vectors_count
        else:
            # In Qdrant 1.16.1+, indexed_vectors_count may not be available
            result["indexed_vectors_count"] = collection_info.points_count

        return result

    # ==================== Point Operations ====================

    def _normalize_point_id(self, point_id: Any) -> int | uuid.UUID:
        """Normalize point ID to integer or UUID for Qdrant 1.16.1+ compatibility.

        Qdrant 1.16.1+ requires point IDs to be either unsigned integers or UUIDs.
        This function converts string IDs (like 'vec-1') to integers or UUIDs.
        """
        # If already int or UUID, return as-is
        if isinstance(point_id, (int, uuid.UUID)):
            return point_id

        # If string, try to convert
        if isinstance(point_id, str):
            # Try to parse as integer first
            try:
                # Handle string IDs like "vec-1" by extracting the number
                if point_id.startswith("vec-"):
                    num_str = point_id.split("-", 1)[-1]
                    return int(num_str)
                # Try direct integer conversion
                return int(point_id)
            except (ValueError, AttributeError):
                # If not a valid integer, try UUID
                try:
                    return uuid.UUID(point_id)
                except (ValueError, AttributeError):
                    # Fallback: generate UUID from string hash
                    return uuid.uuid5(uuid.NAMESPACE_DNS, point_id)

        # For other types, try to convert to int
        try:
            return int(point_id)
        except (ValueError, TypeError):
            # Last resort: generate UUID from string representation
            return uuid.uuid5(uuid.NAMESPACE_DNS, str(point_id))

    def _upsert_points_sync(
        self,
        collection_name: str,
        points: list[PointStruct] | list[dict],
        **kwargs,  # noqa: ARG002
    ):
        """Synchronous upsert for internal use."""
        self._ensure_collection_exists(collection_name)

        # Convert dict to PointStruct if needed, and normalize vectors
        if not points:
            msg = "Points list cannot be empty"
            raise ValueError(msg)
        if isinstance(points[0], dict):
            points = [
                PointStruct(
                    id=self._normalize_point_id(point["id"]),
                    vector=self._normalize_vector(point["vector"]),
                    payload=point.get("payload", {}),
                )
                for point in points
            ]
        else:
            # Normalize vectors and IDs in existing PointStruct objects
            points = [
                PointStruct(
                    id=self._normalize_point_id(point.id),
                    vector=self._normalize_vector(point.vector),
                    payload=point.payload,
                )
                for point in points
            ]

        # Upsert in batches using existing _upsert method
        for points_chunk in chunks(points, batch_size=100):
            self._upsert(collection_name, points_chunk)

    async def upsert(
        self,
        collection_name: str,
        points: list[PointStruct] | list[dict],
        **kwargs,
    ):
        """
        Insert or update points in a collection.

        Args:
            collection_name: Name of the collection
            points: List of PointStruct objects or dictionaries with id, vector, and optional payload
            **kwargs: Additional arguments for upsert operation
        """
        # Use shared synchronous method
        self._upsert_points_sync(collection_name, points, **kwargs)

    async def insert(
        self,
        collection_name: str,
        points: list[PointStruct] | list[dict],
        **kwargs,
    ):
        """
        Insert points into a collection (alias for upsert).

        Args:
            collection_name: Name of the collection
            points: List of PointStruct objects or dictionaries with id, vector, and optional payload
            **kwargs: Additional arguments for insert operation
        """
        await self.upsert(collection_name, points, **kwargs)

    async def delete(
        self,
        collection_name: str,
        points_selector: list[int] | int,
        **kwargs,
    ):
        """
        Delete points from a collection.

        Args:
            collection_name: Name of the collection
            points_selector: Point IDs to delete (single ID or list of IDs)
            **kwargs: Additional arguments for delete operation
        """
        self._ensure_collection_exists(collection_name)

        # Convert single ID to list if needed
        if isinstance(points_selector, int):
            points_selector = [points_selector]

        self.client.delete(
            collection_name=collection_name, points_selector=points_selector, **kwargs
        )

    async def retrieve(
        self,
        collection_name: str,
        ids: list[int] | int,
        with_payload: bool = True,
        with_vectors: bool = False,
        **kwargs,
    ) -> list[dict]:
        """
        Retrieve points by their IDs.

        Args:
            collection_name: Name of the collection
            ids: Point IDs to retrieve (single ID or list of IDs)
            with_payload: Whether to include payload in results
            with_vectors: Whether to include vectors in results
            **kwargs: Additional arguments for retrieve operation

        Returns:
            List of point dictionaries
        """
        self._ensure_collection_exists(collection_name)

        # Convert single ID to list if needed
        if isinstance(ids, int):
            ids = [ids]

        points = self.client.retrieve(
            collection_name=collection_name,
            ids=ids,
            with_payload=with_payload,
            with_vectors=with_vectors,
            **kwargs,
        )

        # Convert to list of dicts for easier use
        result = []
        for point in points:
            point_dict = {"id": point.id}
            if with_payload and point.payload:
                point_dict["payload"] = point.payload
            if with_vectors and point.vector:
                point_dict["vector"] = point.vector
            result.append(point_dict)

        return result

    # ==================== Search Operations ====================

    def _search_sync(
        self,
        collection_name: str,
        query_vector: list[float] | np.ndarray,
        limit: int = 10,
        score_threshold: float | None = None,
        query_filter: Filter | None = None,
        **kwargs,
    ) -> list[ScoredPoint]:
        """Synchronous search for internal use."""
        self._ensure_collection_exists(collection_name)

        # Build kwargs for search
        search_kwargs = {"score_threshold": score_threshold, "query_filter": query_filter, **kwargs}
        # Remove None values
        search_kwargs = {k: v for k, v in search_kwargs.items() if v is not None}

        # Use _query which handles retry logic and vector normalization
        return self._query(collection_name, query_vector, limit, **search_kwargs)

    def _resolve_payload_url(
        self, payload: dict[str, Any], collection_name: str, point_id: Any
    ) -> str:
        source = (
            payload.get("source")
            or payload.get("url")
            or payload.get("file_path")
            or payload.get("path")
        )
        if isinstance(source, str) and source:
            if source.startswith(("http://", "https://", "file://")):
                return source

            source_path = Path(source).expanduser()
            try:
                resolved = source_path.resolve()
                if resolved.exists() or source_path.is_absolute():
                    return resolved.as_uri()
            except Exception:
                return str(source_path)
            return str(source_path)

        return f"qdrant://{collection_name}/{point_id}"

    def _resolve_payload_title(self, payload: dict[str, Any], url: str, page: Any) -> str:
        raw_title = payload.get("title")
        if isinstance(raw_title, str) and raw_title.strip():
            base = raw_title.strip()
        else:
            parsed = urlparse(url)
            path_part = parsed.path or url
            base = Path(path_part).stem or url

        try:
            page_int = int(page) if page is not None else None
        except (TypeError, ValueError):
            page_int = None

        if Path(urlparse(url).path).suffix.lower() == ".pdf" and page_int is not None:
            base = f"{base}#p{page_int}"

        return base

    def _format_search_results(self, points: list[ScoredPoint] | None, collection_name: str):
        results: list[dict[str, Any]] = []

        for point in points or []:
            payload = getattr(point, "payload", {}) or {}
            text = payload.get("text") or payload.get("content")
            if isinstance(text, list):
                text = " ".join([t for t in text if isinstance(t, str)])
            if not isinstance(text, str):
                continue
            excerpt = text.strip()
            if not excerpt:
                continue

            page = payload.get("page") or payload.get("page_number") or payload.get("pageIndex")
            url = self._resolve_payload_url(payload, collection_name, getattr(point, "id", ""))
            title = self._resolve_payload_title(payload, url, page)

            results.append(
                {
                    "url": url,
                    "title": title,
                    "excerpts": [excerpt],
                    "source_id": payload.get("source_id")
                    or payload.get("sourceId")
                    or payload.get("chunk_id")
                    or payload.get("chunkId")
                    or getattr(point, "id", None),
                }
            )

        return SearchResult({"results": results})

    async def search(
        self,
        collection_name: str,
        query_vector: list[float] | np.ndarray,
        limit: int = 10,
        score_threshold: float | None = None,
        query_filter: Filter | None = None,
        **kwargs,
    ) -> list[ScoredPoint]:
        """
        Search for similar vectors in a collection.

        Args:
            collection_name: Name of the collection to search
            query_vector: Query vector to search for
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            query_filter: Optional filter to apply to the search
            **kwargs: Additional search parameters

        Returns:
            List of ScoredPoint objects containing id, score, and payload
        """
        # Use shared synchronous method
        return self._search_sync(
            collection_name, query_vector, limit, score_threshold, query_filter, **kwargs
        )

    # ==================== Document Operations with Chunking ====================

    def _download_and_read_file(self, file_url: str) -> str:
        """
        Download file from URL and read it using FileReader.

        Args:
            file_url: URL to the file to download

        Returns:
            Text content of the file
        """
        if self.reader is None:
            msg = "FileReader not initialized"
            raise RuntimeError(msg)

        file_path = Path(file_url)
        suffix = file_path.suffix
        with (
            urllib.request.urlopen(file_url) as f,
            tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file,
        ):
            tmp_file.write(f.read())
            tmp_file.flush()
            tmp_file_name = tmp_file.name

        try:
            content = self.reader(tmp_file_name)
            return content.value[0] if isinstance(content.value, list) else str(content.value)
        finally:
            # Clean up temporary file
            tmp_path = Path(tmp_file_name)
            if tmp_path.exists():
                tmp_path.unlink()

    async def chunk_and_upsert(
        self,
        collection_name: str,
        text: str | Symbol | None = None,
        document_path: str | None = None,
        document_url: str | None = None,
        chunker_name: str | None = None,
        chunker_kwargs: dict | None = None,
        start_id: int | None = None,
        metadata: dict | None = None,
        **upsert_kwargs,
    ):
        """
        Chunk text or documents using ChonkieChunker and upsert the chunks with embeddings into Qdrant.

        Args:
            collection_name: Name of the collection to upsert into
            text: Text to chunk (string or Symbol). If None, document_path or document_url must be provided.
            document_path: Path to a document file to read using FileReader (PDF, etc.)
            document_url: URL to a document file to download and read using FileReader
            chunker_name: Name of the chunker to use. If None, uses the instance default chunker_name
            chunker_kwargs: Additional keyword arguments for the chunker
            start_id: Starting ID for the chunks (auto-incremented). If None, uses hash-based IDs
            metadata: Optional metadata to add to all chunk payloads
            **upsert_kwargs: Additional arguments for upsert operation

        Returns:
            Number of chunks upserted
        """
        self._ensure_collection_exists(collection_name)

        # Validate input: exactly one of text, document_path, or document_url must be provided
        input_count = sum(x is not None for x in [text, document_path, document_url])
        if input_count == 0:
            msg = "One of `text`, `document_path`, or `document_url` must be provided"
            raise ValueError(msg)
        if input_count > 1:
            msg = "Only one of `text`, `document_path`, or `document_url` can be provided"
            raise ValueError(msg)

        # Get collection info to determine vector size
        collection_info = await self.get_collection_info(collection_name)
        vector_config = collection_info["config"]["params"]["vectors"]
        if "size" in vector_config:
            vector_size = vector_config["size"]
        else:
            # For named vectors, we need to specify which one to use
            # Default to first named vector or raise error
            named_vectors = vector_config.get("named_vectors", {})
            if not named_vectors:
                msg = "Collection has no vector configuration"
                raise ValueError(msg)
            vector_size = next(iter(named_vectors.values()))["size"]

        # Check if chunker is initialized
        if self.chunker is None:
            msg = "Chunker not initialized. Please ensure ChonkieChunker is available."
            raise RuntimeError(msg)

        # Use instance chunker and default chunker_name if not provided
        chunker_kwargs = chunker_kwargs or {}
        if chunker_name is None:
            chunker_name = self.chunker_name

        # Handle document_path: read file using FileReader
        if document_path is not None:
            if self.reader is None:
                msg = "FileReader not initialized. Please ensure FileReader is available."
                raise RuntimeError(msg)
            doc_path = Path(document_path)
            if not doc_path.exists():
                msg = f"Document file not found: {document_path}"
                raise FileNotFoundError(msg)
            content = self.reader(document_path)
            text = content.value[0] if isinstance(content.value, list) else str(content.value)
            # Add source to metadata if not already present
            if metadata is None:
                metadata = {}
            metadata["source"] = str(doc_path.resolve())

        # Handle document_url: download and read file using FileReader
        elif document_url is not None:
            if self.reader is None:
                msg = "FileReader not initialized. Please ensure FileReader is available."
                raise RuntimeError(msg)
            text = self._download_and_read_file(document_url)
            # Add source to metadata if not already present
            if metadata is None:
                metadata = {}
            if "source" not in metadata:
                metadata["source"] = document_url

        # Convert text to Symbol if needed
        text_symbol = Symbol(text) if isinstance(text, str) else text

        # Chunk the text using instance chunker
        chunks_symbol = self.chunker.forward(
            text_symbol, chunker_name=chunker_name, **chunker_kwargs
        )
        chunks = chunks_symbol.value if hasattr(chunks_symbol, "value") else chunks_symbol

        if not chunks:
            warnings.warn("No chunks generated from text")
            return 0

        # Ensure chunks is a list
        if not isinstance(chunks, list):
            chunks = [chunks]

        # Generate embeddings and create points
        points = []
        current_id = start_id if start_id is not None else 0
        for chunk_item in chunks:
            # Clean the chunk text
            if ChonkieChunker:
                chunk_text = ChonkieChunker.clean_text(str(chunk_item))
            else:
                chunk_text = str(chunk_item)

            if not chunk_text.strip():
                continue

            # Generate embedding using Symbol's embedding property
            chunk_symbol = Symbol(chunk_text)

            # Generate embedding - Symbol has embedding property that returns numpy array
            try:
                embedding = chunk_symbol.embedding
            except (AttributeError, Exception) as e:
                # Fallback: try using Expression's embed method
                try:
                    embedding = chunk_symbol.embed()
                    if hasattr(embedding, "value"):
                        embedding = embedding.value
                except Exception as embed_err:
                    msg = f"Could not generate embedding for chunk. Error: {e}"
                    raise ValueError(msg) from embed_err

            # Normalize embedding to flat list using existing helper
            if isinstance(embedding, np.ndarray):
                # Flatten if 2D (e.g., shape (1, 1536) -> (1536,))
                if embedding.ndim > 1:
                    embedding = embedding.flatten()
                embedding = embedding.tolist()
            elif isinstance(embedding, list):
                # Ensure embedding is a flat list (handle nested lists)
                if embedding and len(embedding) > 0 and isinstance(embedding[0], list):
                    # Flatten nested list (e.g., [[1, 2, 3]] -> [1, 2, 3])
                    embedding = (
                        embedding[0]
                        if len(embedding) == 1
                        else [item for sublist in embedding for item in sublist]
                    )
            else:
                # Try to convert to list
                try:
                    embedding = list(embedding) if embedding else []
                except (TypeError, ValueError) as e:
                    msg = (
                        f"Could not generate embedding for chunk. "
                        f"Expected list or array, got type: {type(embedding)}"
                    )
                    raise ValueError(msg) from e

            # Truncate or pad embedding to match vector_size
            original_size = len(embedding)
            if original_size != vector_size:
                if original_size > vector_size:
                    embedding = embedding[:vector_size]
                else:
                    embedding = embedding + [0.0] * (vector_size - original_size)
                warnings.warn(
                    f"Embedding size ({original_size}) adjusted to match collection vector size ({vector_size})"
                )

            # Create payload
            payload = {"text": chunk_text}
            if metadata:
                payload.update(metadata)

            # Generate ID
            if start_id is not None:
                point_id = current_id
                current_id += 1
            else:
                # Use uuid5 for deterministic, collision-resistant IDs based on content
                # uuid5 uses SHA-1 internally, providing 160 bits of entropy
                # Convert to int64 by taking modulo 2**63 to fit in signed 64-bit range
                namespace_uuid = uuid.NAMESPACE_DNS  # Use DNS namespace for consistency
                uuid_obj = uuid.uuid5(namespace_uuid, chunk_text)
                # Convert UUID (128 bits) to int64, ensuring it fits in signed 64-bit range
                point_id = uuid_obj.int % (2**63)

            points.append(
                {
                    "id": point_id,
                    "vector": embedding,
                    "payload": payload,
                }
            )

        if not points:
            warnings.warn("No valid points to upsert")
            return 0

        # Upsert the points using shared synchronous method
        self._upsert_points_sync(collection_name=collection_name, points=points, **upsert_kwargs)

        return len(points)
