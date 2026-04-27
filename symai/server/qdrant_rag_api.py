from __future__ import annotations

import asyncio
import os
import resource
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlparse, urlunparse

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from symai.backend.engines.index.engine_qdrant import QdrantIndexEngine, chunks
from symai.core import Argument
from symai.symbol import Symbol as SymaiSymbol
from symai.utils import UserMessage

try:
    from qdrant_client.http import models
    from qdrant_client.http.models import Filter
except ImportError:
    models = None
    Filter = None

_STARTUP_TIME = time.monotonic()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        msg = f"Environment variable {name} must be an integer"
        raise RuntimeError(msg) from exc


def _env_str(name: str, default: str | None = None) -> str | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw


class ApiSettings(BaseModel):
    # Qdrant connection (prefer SYMAI_QDRANT_URL, fallback to INDEXING_ENGINE_URL)
    qdrant_url: str | None = Field(
        default_factory=lambda: _env_str("SYMAI_QDRANT_URL", _env_str("INDEXING_ENGINE_URL"))
    )
    qdrant_api_key: str | None = Field(default_factory=lambda: _env_str("INDEXING_ENGINE_API_KEY"))

    # Default collection/index settings
    index_name: str = Field(default_factory=lambda: os.getenv("SYMAI_INDEX_NAME", "dataindex"))
    index_dims: int = Field(default_factory=lambda: _env_int("SYMAI_INDEX_DIMS", 1536))
    index_top_k: int = Field(default_factory=lambda: _env_int("SYMAI_INDEX_TOP_K", 5))
    index_metric: str = Field(default_factory=lambda: os.getenv("SYMAI_INDEX_METRIC", "Cosine"))
    tokenizer_name: str = Field(default_factory=lambda: os.getenv("SYMAI_TOKENIZER", "gpt2"))
    embedding_model_name: str = Field(
        default_factory=lambda: os.getenv("SYMAI_EMBED_MODEL", "minishlab/potion-base-8M")
    )

    # API auth
    rag_api_token: str | None = Field(default_factory=lambda: _env_str("RAG_API_TOKEN"))

    # Upload directory for document_path resolution
    rag_upload_dir: str | None = Field(default_factory=lambda: _env_str("RAG_UPLOAD_DIR"))


SETTINGS = ApiSettings()

if not SETTINGS.rag_api_token and os.getenv("RAG_ALLOW_NO_TOKEN") != "1":
    logger.critical(
        "RAG_API_TOKEN is not set. The API will accept unauthenticated requests. "
        "Set RAG_API_TOKEN or RAG_ALLOW_NO_TOKEN=1 to proceed."
    )


class VectorPoint(BaseModel):
    id: int | str | uuid.UUID
    vector: list[float] = Field(..., min_items=1)
    payload: dict[str, Any] | None = None

    @field_validator("id", mode="before")
    @classmethod
    def _convert_id(cls, value: Any) -> int | uuid.UUID:
        """Convert ID to valid Qdrant format: unsigned int or UUID.

        Matches engine's _normalize_point_id logic:
        - Handles "vec-1" style IDs by extracting the number part
        - Converts other strings to UUID if not parseable as int
        """
        if isinstance(value, uuid.UUID):
            return value
        if isinstance(value, int):
            if value < 0:
                msg = "Point ID must be a non-negative integer or UUID"
                raise ValueError(msg)
            return value
        if isinstance(value, str):
            if value.startswith("vec-"):
                try:
                    num_str = value.split("-", 1)[-1]
                    int_id = int(num_str)
                    if int_id < 0:
                        msg = "Point ID must be a non-negative integer or UUID"
                        raise ValueError(msg)
                    return int_id
                except (ValueError, AttributeError):
                    pass
            try:
                int_id = int(value)
                if int_id < 0:
                    msg = "Point ID must be a non-negative integer or UUID"
                    raise ValueError(msg)
                return int_id
            except ValueError:
                try:
                    return uuid.UUID(value)
                except (ValueError, AttributeError):
                    return uuid.uuid5(uuid.NAMESPACE_DNS, value)
        msg = f"Point ID must be a non-negative integer, UUID, or string (got {type(value)})"
        raise ValueError(msg)

    @field_validator("vector")
    @classmethod
    def _flatten_vectors(cls, value: Sequence[float]) -> list[float]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            msg = "vector must be a numeric sequence"
            raise ValueError(msg)
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError) as exc:
            msg = "vector must contain numeric values"
            raise ValueError(msg) from exc


class PointsUpsertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_name: str | None = Field(default=None, description="Qdrant collection name.")
    points: list[VectorPoint] = Field(..., min_items=1, max_items=10_000)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_name: str | None = None
    query: str | None = Field(
        default=None, description="Plain-text query embedded via SymbolicAI.", max_length=100_000
    )
    embedding: list[float] | None = Field(
        default=None, description="Custom embedding vector.", max_length=16_384
    )
    limit: int | None = Field(default=None, ge=1, le=10_000)
    score_threshold: float | None = None
    filter: dict[str, Any] | None = None
    with_payload: bool = True
    with_vectors: bool = False

    @model_validator(mode="before")
    @classmethod
    def _require_query_or_embedding(cls, values: Any) -> Any:
        if isinstance(values, dict) and not values.get("query") and not values.get("embedding"):
            msg = "Either `query` or `embedding` must be provided."
            raise ValueError(msg)
        return values

    @field_validator("embedding")
    @classmethod
    def _normalize_embedding(cls, value: Sequence[float] | None) -> list[float] | None:
        if value is None:
            return None
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            msg = "embedding must be a numeric sequence"
            raise ValueError(msg)
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError) as exc:
            msg = "embedding must contain numeric values"
            raise ValueError(msg) from exc


class SearchMatch(BaseModel):
    id: int | str | None = None
    score: float
    payload: dict[str, Any] | None = None
    vector: list[float] | None = None


class SearchResponse(BaseModel):
    matches: list[SearchMatch]


class ChunkUpsertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_name: str | None = None
    text: str | None = Field(default=None, max_length=10_000_000)
    document_path: str | None = Field(default=None, max_length=4096)
    document_url: str | None = Field(default=None, max_length=4096)
    metadata: dict[str, Any] | None = None
    chunker_name: str | None = None
    chunker_kwargs: dict[str, Any] | None = None
    start_id: int | None = None
    embed_batch_size: int = Field(
        default_factory=lambda: _env_int("RAG_EMBED_BATCH_SIZE", 100)
    )

    @model_validator(mode="before")
    @classmethod
    def _validate_payload(cls, values: Any) -> Any:
        if isinstance(values, dict) and not any(
            values.get(k) for k in ("text", "document_path", "document_url")
        ):
            msg = "Provide one of text, document_path, or document_url."
            raise ValueError(msg)
        return values


class BatchChunkItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    metadata: dict[str, Any] | None = None
    skip_chunking: bool = Field(default=False, description="If True, treat text as a single chunk (skip splitting).")


class BatchChunkUpsertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_name: str | None = None
    items: list[BatchChunkItem] = Field(..., min_length=1, max_length=10_000)
    chunker_name: str | None = None
    chunker_kwargs: dict[str, Any] | None = None
    embed_batch_size: int = Field(
        default_factory=lambda: _env_int("RAG_EMBED_BATCH_SIZE", 100)
    )


class CollectionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    vector_size: int | None = None
    distance: str | None = None


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_name: str | None = None
    ids: list[int | str | uuid.UUID] = Field(..., min_items=1)
    with_payload: bool = True
    with_vectors: bool = False

    @field_validator("ids", mode="before")
    @classmethod
    def _convert_ids(cls, value: Any) -> list[int | uuid.UUID]:
        if not isinstance(value, list):
            msg = "ids must be a list"
            raise ValueError(msg)
        return [VectorPoint._convert_id(v) for v in value]


class DeletePointsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_name: str | None = None
    ids: list[int | str | uuid.UUID] = Field(..., min_items=1)

    @field_validator("ids", mode="before")
    @classmethod
    def _convert_ids(cls, value: Any) -> list[int | uuid.UUID]:
        if not isinstance(value, list):
            msg = "ids must be a list"
            raise ValueError(msg)
        return [VectorPoint._convert_id(v) for v in value]


class MetadataFilterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_name: str | None = None
    metadata: dict[str, Any]
    exact: bool = True

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict) or not value:
            msg = "metadata must be a non-empty object"
            raise ValueError(msg)
        return value


class MetadataListRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_name: str | None = None
    metadata: dict[str, Any]
    batch_size: int = Field(default=256, ge=1, le=4096)
    max_points: int = Field(default=20000, ge=1, le=100000)
    with_payload: bool = True
    with_vectors: bool = False

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict) or not value:
            msg = "metadata must be a non-empty object"
            raise ValueError(msg)
        return value


class DeleteByTenantFileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_name: str | None = None
    source_type: str = "gdrive"
    file_id: str = Field(..., min_length=1)


class DeleteByTenantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_name: str | None = None
    source_type: str = "gdrive"


app = FastAPI(
    title="SymbolicAI Qdrant RAG API",
    version="0.1.0",
    description="FastAPI wrapper exposing Qdrant-backed indexing utilities.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next: Any) -> Response:
    t0 = time.monotonic()
    response = await call_next(request)
    duration_ms = (time.monotonic() - t0) * 1000
    path = request.url.path
    method = request.method
    status_code = response.status_code
    log_msg = "RAG {} {} -> {} ({:.0f}ms)"
    if duration_ms > 5000:
        logger.warning(log_msg, method, path, status_code, duration_ms)
    elif status_code >= 400:
        logger.error(log_msg, method, path, status_code, duration_ms)
    else:
        logger.info(log_msg, method, path, status_code, duration_ms)
    return response


def _rss_mb() -> float:
    """Return current process RSS in megabytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is in KB on Linux, bytes on macOS
    if os.uname().sysname == "Darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


ENGINE: QdrantIndexEngine | None = None
ENGINE_LOCK = Lock()


def _ensure_symbolicai_available() -> None:
    if QdrantIndexEngine is None:  # pragma: no cover - depends on runtime env
        msg = (
            "SymbolicAI is not installed. Install the `symbolicai[qdrant]` extra before "
            "running the API."
        )
        raise RuntimeError(msg)


def _build_engine() -> QdrantIndexEngine:
    _ensure_symbolicai_available()
    assert QdrantIndexEngine is not None
    return QdrantIndexEngine(
        url=SETTINGS.qdrant_url,
        api_key=SETTINGS.qdrant_api_key,
        index_name=SETTINGS.index_name,
        index_dims=SETTINGS.index_dims,
        index_top_k=SETTINGS.index_top_k,
        index_metric=SETTINGS.index_metric,
        tokenizer_name=SETTINGS.tokenizer_name,
        embedding_model_name=SETTINGS.embedding_model_name,
        upload_dir=SETTINGS.rag_upload_dir,
    )


async def get_engine() -> QdrantIndexEngine:
    global ENGINE
    if ENGINE is not None:
        return ENGINE
    with ENGINE_LOCK:
        if ENGINE is None:
            ENGINE = _build_engine()
    assert ENGINE is not None
    return ENGINE


def _resolve_index_name(name: str | None) -> str:
    return name or SETTINGS.index_name


def _translate_document_path(path: str) -> str:
    """Translate host file paths to container paths used by extensity-rag."""
    if not path:
        return path

    project_name = "Qdrant-symai-server-api"
    container_root = Path("/opt/qdrant-rag")
    if path.startswith(str(container_root)):
        return path
    if project_name in path:
        idx = path.find(project_name)
        if idx != -1:
            relative_part = path[idx + len(project_name) :].lstrip("/")
            translated = (container_root / relative_part).resolve()
            if not str(translated).startswith(str(container_root)):
                UserMessage(
                    f"Path traversal detected: '{path}' resolves outside container root",
                    raise_with=ValueError,
                )
            translated_str = str(translated)
            logger.debug("Translated path: {} -> {}", path, translated_str)
            return translated_str
    return path


def _serialize_point(point: Any) -> dict[str, Any]:
    if isinstance(point, dict):
        resp: dict[str, Any] = {
            "id": point.get("id"),
            "payload": point.get("payload"),
            "vector": point.get("vector"),
        }
        score = point.get("score")
        if score is not None:
            resp["score"] = float(score)
        return resp

    payload = getattr(point, "payload", None)
    vector = getattr(point, "vector", None)
    score = getattr(point, "score", None)
    resp = {"id": getattr(point, "id", None), "payload": payload, "vector": vector}
    if score is not None:
        resp["score"] = float(score)
    return resp


def _embedding_from_query(query: str) -> list[float]:
    symbol_cls = SymaiSymbol
    if symbol_cls is None:  # pragma: no cover
        detail = "SymbolicAI is not installed; cannot transform queries into embeddings."
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        )

    symbol = symbol_cls(query)
    embedding = getattr(symbol, "embedding", None)
    if embedding is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SymbolicAI returned an empty embedding for the supplied query.",
        )

    if Argument is not None and isinstance(embedding, Argument):
        embedding = getattr(embedding, "value", embedding)
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    if not isinstance(embedding, Sequence) or isinstance(embedding, (str, bytes)):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SymbolicAI produced an unsupported embedding payload.",
        )

    if (
        embedding
        and isinstance(embedding[0], Sequence)
        and not isinstance(embedding[0], (str, bytes))
    ):
        if len(embedding) == 1:
            embedding = embedding[0]
        else:
            embedding = [item for sub in embedding for item in sub]

    try:
        return [float(v) for v in embedding]
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SymbolicAI embeddings must be numeric.",
        ) from exc


async def require_token(
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    expected = SETTINGS.rag_api_token
    if not expected:
        return

    provided: str | None = None
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer":
            provided = token.strip()
        elif not token:
            provided = scheme.strip()
    if not provided and x_api_key:
        provided = x_api_key.strip()

    if provided != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_tenant(
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
) -> str:
    """Extract tenant_id from the trusted X-Tenant-Id header.

    This dependency runs AFTER `require_token` has validated the caller's
    API key. Because only first-party services hold valid API keys, the
    header value is authoritative once authentication passes.
    """
    if x_tenant_id is None or not x_tenant_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing or empty X-Tenant-Id header.",
        )
    return x_tenant_id


@app.get("/", dependencies=[Depends(require_token)])
async def root() -> dict[str, Any]:
    return {
        "service": "SymbolicAI Qdrant RAG",
        "docs": "/docs",
        "default_index": SETTINGS.index_name,
    }


@app.get("/healthz", dependencies=[Depends(require_token)])
async def healthz(engine: QdrantIndexEngine = Depends(get_engine)) -> dict[str, Any]:  # noqa: B008
    collection_stats: dict[str, Any] = {}
    try:
        collections = await engine.list_collections()
        ready = True
        for coll_name in collections:
            try:
                info = await engine.get_collection_info(collection_name=coll_name)
                if isinstance(info, dict):
                    collection_stats[coll_name] = {
                        "points_count": info.get("points_count"),
                        "vectors_count": info.get("vectors_count"),
                        "indexed_vectors_count": info.get("indexed_vectors_count"),
                    }
                else:
                    points_count = getattr(info, "points_count", None)
                    vectors_count = getattr(info, "vectors_count", None)
                    indexed = getattr(info, "indexed_vectors_count", None)
                    collection_stats[coll_name] = {
                        "points_count": points_count,
                        "vectors_count": vectors_count,
                        "indexed_vectors_count": indexed,
                    }
            except Exception:
                collection_stats[coll_name] = {"error": "failed to fetch stats"}
    except Exception as exc:  # pragma: no cover
        logger.exception("Health check failed while listing collections: {}", exc)
        collections = []
        ready = False
    parsed = urlparse(SETTINGS.qdrant_url or "")
    redacted_qdrant_url = (
        urlunparse(parsed._replace(netloc="[REDACTED]"))
        if parsed.password
        else SETTINGS.qdrant_url
    )
    return {
        "status": "ok" if ready else "degraded",
        "ready": ready,
        "collections": collections,
        "collection_stats": collection_stats,
        "index_name": SETTINGS.index_name,
        "qdrant_url": redacted_qdrant_url,
        "token_required": bool(SETTINGS.rag_api_token),
        "rss_mb": round(_rss_mb(), 1),
        "uptime_seconds": round(time.monotonic() - _STARTUP_TIME, 1),
    }


@app.post("/points", dependencies=[Depends(require_token), Depends(require_tenant)])
async def upsert_points(
    payload: PointsUpsertRequest,
    tenant_id: str = Depends(require_tenant),
    engine: QdrantIndexEngine = Depends(get_engine),  # noqa: B008
) -> dict[str, Any]:
    collection = _resolve_index_name(payload.index_name)
    points_data: list[dict[str, Any]] = []
    for point in payload.points:
        point_dict = point.model_dump(mode="python")
        if isinstance(point_dict.get("id"), uuid.UUID):
            point_dict["id"] = str(point_dict["id"])
        points_data.append(point_dict)
    t0 = time.monotonic()
    await engine.upsert(
        collection_name=collection, points=points_data, tenant_id=tenant_id
    )
    duration_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "upsert collection={} points={} duration={:.0f}ms",
        collection, len(payload.points), duration_ms,
    )
    return {"points_upserted": len(payload.points), "collection": collection}


@app.delete("/points", dependencies=[Depends(require_token), Depends(require_tenant)])
async def delete_points(
    payload: DeletePointsRequest,
    tenant_id: str = Depends(require_tenant),
    engine: QdrantIndexEngine = Depends(get_engine),  # noqa: B008
) -> dict[str, Any]:
    collection = _resolve_index_name(payload.index_name)
    ids = [str(v) if isinstance(v, uuid.UUID) else v for v in payload.ids]
    await engine.delete(
        collection_name=collection, points_selector=ids, tenant_id=tenant_id
    )
    return {"points_deleted": len(ids), "collection": collection}


@app.post("/points/count-by-metadata", dependencies=[Depends(require_token), Depends(require_tenant)])
async def count_points_by_metadata(
    payload: MetadataFilterRequest,
    tenant_id: str = Depends(require_tenant),
    engine: QdrantIndexEngine = Depends(get_engine),  # noqa: B008
) -> dict[str, Any]:
    collection = _resolve_index_name(payload.index_name)
    count = await engine.count(
        collection_name=collection,
        query_filter=payload.metadata,
        exact=payload.exact,
        tenant_id=tenant_id,
    )
    return {"count": int(count), "collection": collection}


@app.post("/points/delete-by-metadata", dependencies=[Depends(require_token), Depends(require_tenant)])
async def delete_points_by_metadata(
    payload: MetadataFilterRequest,
    tenant_id: str = Depends(require_tenant),
    engine: QdrantIndexEngine = Depends(get_engine),  # noqa: B008
) -> dict[str, Any]:
    collection = _resolve_index_name(payload.index_name)
    count = await engine.count(
        collection_name=collection,
        query_filter=payload.metadata,
        exact=payload.exact,
        tenant_id=tenant_id,
    )
    t0 = time.monotonic()
    await engine.delete_by_filter(
        collection_name=collection,
        query_filter=payload.metadata,
        tenant_id=tenant_id,
    )
    duration_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "delete-by-metadata collection={} points_deleted={} duration={:.0f}ms",
        collection, int(count), duration_ms,
    )
    return {"points_deleted": int(count), "collection": collection}


@app.post("/points/list-by-metadata", dependencies=[Depends(require_token), Depends(require_tenant)])
async def list_points_by_metadata(
    payload: MetadataListRequest,
    tenant_id: str = Depends(require_tenant),
    engine: QdrantIndexEngine = Depends(get_engine),  # noqa: B008
) -> dict[str, Any]:
    collection = _resolve_index_name(payload.index_name)
    await asyncio.to_thread(engine._ensure_collection_exists, collection)
    built_filter = engine._build_query_filter(payload.metadata, tenant_id=tenant_id)
    if built_filter is None:
        detail = "metadata filter must not be empty"
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
        )

    points: list[dict[str, Any]] = []
    offset: Any | None = None
    while len(points) < payload.max_points:
        records, next_offset = await asyncio.to_thread(
            engine.client.scroll,
            collection_name=collection,
            scroll_filter=built_filter,
            limit=payload.batch_size,
            offset=offset,
            with_payload=payload.with_payload,
            with_vectors=payload.with_vectors,
        )
        if not records:
            break
        for record in records:
            serialized = _serialize_point(record)
            if not payload.with_vectors:
                serialized["vector"] = None
            points.append(serialized)
            if len(points) >= payload.max_points:
                break
        if len(points) >= payload.max_points or next_offset is None:
            break
        offset = next_offset

    return {"points": points, "collection": collection}


@app.post(
    "/delete-by-tenant-file",
    dependencies=[Depends(require_token), Depends(require_tenant)],
)
async def delete_by_tenant_file(
    payload: DeleteByTenantFileRequest,
    tenant_id: str = Depends(require_tenant),
    engine: QdrantIndexEngine = Depends(get_engine),  # noqa: B008
) -> dict[str, Any]:
    collection = _resolve_index_name(payload.index_name)
    await asyncio.to_thread(engine._ensure_collection_exists, collection)
    await asyncio.to_thread(engine._ensure_payload_indexes, collection)

    if models is None or Filter is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Qdrant client models not available.",
        )

    query_filter = Filter(
        must=[
            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
            models.FieldCondition(
                key="source_type", match=models.MatchValue(value=payload.source_type)
            ),
            models.FieldCondition(key="file_id", match=models.MatchValue(value=payload.file_id)),
        ]
    )

    count = await engine.count(
        collection_name=collection,
        query_filter=query_filter,
        exact=True,
        tenant_id=tenant_id,
    )
    t0 = time.monotonic()
    await engine.delete_by_filter(
        collection_name=collection,
        query_filter=query_filter,
        tenant_id=tenant_id,
    )
    duration_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "delete-by-tenant-file collection={} tenant={} file_id={} points_deleted={} duration={:.0f}ms",
        collection, tenant_id, payload.file_id, int(count), duration_ms,
    )
    return {"points_deleted": int(count), "collection": collection}


@app.post(
    "/delete-by-tenant",
    dependencies=[Depends(require_token), Depends(require_tenant)],
)
async def delete_by_tenant(
    payload: DeleteByTenantRequest,
    tenant_id: str = Depends(require_tenant),
    engine: QdrantIndexEngine = Depends(get_engine),  # noqa: B008
) -> dict[str, Any]:
    collection = _resolve_index_name(payload.index_name)
    await asyncio.to_thread(engine._ensure_collection_exists, collection)
    await asyncio.to_thread(engine._ensure_payload_indexes, collection)

    if models is None or Filter is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Qdrant client models not available.",
        )

    query_filter = Filter(
        must=[
            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
            models.FieldCondition(
                key="source_type", match=models.MatchValue(value=payload.source_type)
            ),
        ]
    )

    count = await engine.count(
        collection_name=collection,
        query_filter=query_filter,
        exact=True,
        tenant_id=tenant_id,
    )
    t0 = time.monotonic()
    await engine.delete_by_filter(
        collection_name=collection,
        query_filter=query_filter,
        tenant_id=tenant_id,
    )
    duration_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "delete-by-tenant collection={} tenant={} points_deleted={} duration={:.0f}ms",
        collection, tenant_id, int(count), duration_ms,
    )
    return {"points_deleted": int(count), "collection": collection}


@app.post("/retrieve", dependencies=[Depends(require_token), Depends(require_tenant)])
async def retrieve_points(
    payload: RetrieveRequest,
    tenant_id: str = Depends(require_tenant),
    engine: QdrantIndexEngine = Depends(get_engine),  # noqa: B008
) -> dict[str, Any]:
    collection = _resolve_index_name(payload.index_name)
    results = await engine.retrieve(
        collection_name=collection,
        ids=payload.ids,
        with_payload=payload.with_payload,
        with_vectors=payload.with_vectors,
        tenant_id=tenant_id,
    )
    return {"collection": collection, "points": [_serialize_point(p) for p in results]}


@app.post("/search", response_model=SearchResponse, dependencies=[Depends(require_token), Depends(require_tenant)])
async def search(
    payload: SearchRequest,
    tenant_id: str = Depends(require_tenant),
    engine: QdrantIndexEngine = Depends(get_engine),  # noqa: B008
) -> SearchResponse:
    collection = _resolve_index_name(payload.index_name)

    t_embed = time.monotonic()
    vector = payload.embedding or _embedding_from_query(payload.query or "")
    embed_ms = (time.monotonic() - t_embed) * 1000

    expected_dim = SETTINGS.index_dims
    if isinstance(vector, Sequence) and not isinstance(vector, (str, bytes)):
        try:
            vector = [float(v) for v in vector]
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Search embedding must contain numeric values.",
            ) from exc

        original_size = len(vector)
        if original_size != expected_dim:
            if original_size > expected_dim:
                vector = vector[:expected_dim]
            else:
                vector = vector + [0.0] * (expected_dim - original_size)

    query_filter = None
    if payload.filter:
        query_filter = payload.filter

    t_search = time.monotonic()
    results = await engine.search(
        collection_name=collection,
        query_vector=vector,
        limit=payload.limit or SETTINGS.index_top_k,
        score_threshold=payload.score_threshold,
        query_filter=query_filter,
        tenant_id=tenant_id,
    )
    search_ms = (time.monotonic() - t_search) * 1000

    matches: list[SearchMatch] = []
    for match in results:
        serialized = _serialize_point(match)
        matches.append(
            SearchMatch(
                id=serialized.get("id"),
                score=float(serialized.get("score", 0.0)),
                payload=serialized.get("payload"),
                vector=serialized.get("vector") if payload.with_vectors else None,
            )
        )

    logger.info(
        "search collection={} embed={:.0f}ms qdrant={:.0f}ms matches={}",
        collection, embed_ms, search_ms, len(matches),
    )
    return SearchResponse(matches=matches)


@app.post("/chunk-upsert", dependencies=[Depends(require_token), Depends(require_tenant)])
async def chunk_and_upsert(
    payload: ChunkUpsertRequest,
    tenant_id: str = Depends(require_tenant),
    engine: QdrantIndexEngine = Depends(get_engine),  # noqa: B008
) -> dict[str, Any]:
    collection = _resolve_index_name(payload.index_name)
    kwargs: dict[str, Any] = {"collection_name": collection}
    if payload.text:
        kwargs["text"] = payload.text
    if payload.document_path:
        kwargs["document_path"] = _translate_document_path(payload.document_path)
    if payload.document_url:
        kwargs["document_url"] = payload.document_url
    if payload.metadata:
        kwargs["metadata"] = payload.metadata
    if payload.chunker_name:
        kwargs["chunker_name"] = payload.chunker_name
    if payload.chunker_kwargs:
        kwargs["chunker_kwargs"] = payload.chunker_kwargs
    if payload.start_id is not None:
        kwargs["start_id"] = payload.start_id
    kwargs["tenant_id"] = tenant_id
    kwargs["embed_batch_size"] = payload.embed_batch_size

    text_len = len(payload.text) if payload.text else 0
    t0 = time.monotonic()
    chunks_indexed = await engine.chunk_and_upsert(**kwargs)
    duration_ms = (time.monotonic() - t0) * 1000

    logger.info(
        "chunk-upsert collection={} chunks={} text_len={} duration={:.0f}ms",
        collection, chunks_indexed, text_len, duration_ms,
    )
    return {"chunks_indexed": chunks_indexed, "collection": collection}


@app.post("/batch-chunk-upsert", dependencies=[Depends(require_token), Depends(require_tenant)])
async def batch_chunk_and_upsert(
    payload: BatchChunkUpsertRequest,
    tenant_id: str = Depends(require_tenant),
    engine: QdrantIndexEngine = Depends(get_engine),  # noqa: B008
) -> dict[str, Any]:
    collection = _resolve_index_name(payload.index_name)
    await asyncio.to_thread(engine._ensure_payload_indexes, collection)
    t0 = time.monotonic()

    chunker_name = payload.chunker_name
    chunker_kwargs = payload.chunker_kwargs or {}

    # Phase 1: Build list of (text, metadata) segments — chunk items that need chunking,
    # pass through items that don't.
    segments = []  # (text, payload)

    t_chunk = time.monotonic()
    seg_idx = 0
    for item in payload.items:
        text = (item.text or "").strip()
        if not text:
            continue
        base_meta = dict(item.metadata or {})
        base_meta["tenant_id"] = tenant_id

        if item.skip_chunking:
            payload_dict = {"text": text, **base_meta}
            segments.append((text, payload_dict))
            seg_idx += 1
        else:
            # Chunk using the engine's chunker
            if engine.chunker is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Chunker not initialized on the engine.",
                )
            text_symbol = SymaiSymbol(text)
            cn = chunker_name if chunker_name is not None else engine.chunker_name
            chunks_symbol = engine.chunker.forward(text_symbol, chunker_name=cn, **chunker_kwargs)
            chunk_items = chunks_symbol.value if hasattr(chunks_symbol, "value") else chunks_symbol
            if not chunk_items:
                continue
            if not isinstance(chunk_items, list):
                chunk_items = [chunk_items]
            for chunk_idx, chunk_item in enumerate(chunk_items):
                chunk_text = str(chunk_item).strip()
                if not chunk_text:
                    continue
                payload_dict = {"text": chunk_text, **base_meta, "chunk_index": chunk_idx}
                segments.append((chunk_text, payload_dict))
                seg_idx += 1
    chunk_ms = (time.monotonic() - t_chunk) * 1000

    if not segments:
        logger.info("batch-chunk-upsert collection={} segments=0 (empty)", collection)
        return {"chunks_indexed": 0, "collection": collection}

    # Phase 2: Batch embed all segments in sized chunks sequentially.
    t_embed = time.monotonic()
    all_texts = [text for text, _ in segments]
    raw_embeddings = []
    for batch in chunks(all_texts, payload.embed_batch_size):
        batch_emb = SymaiSymbol(batch).embed().value
        if Argument is not None and isinstance(batch_emb, Argument):
            batch_emb = getattr(batch_emb, "value", batch_emb)
        if hasattr(batch_emb, "tolist"):
            batch_emb = batch_emb.tolist()
        raw_embeddings.extend(batch_emb)
    embed_ms = (time.monotonic() - t_embed) * 1000

    # Phase 3: Build points with deterministic IDs.
    vector_size = SETTINGS.index_dims
    points = []
    for seg_idx, ((text, payload_dict), raw_vec) in enumerate(
        zip(segments, raw_embeddings, strict=True)
    ):
        vec = engine._normalize_vector(raw_vec)
        if len(vec) != vector_size:
            if len(vec) > vector_size:
                vec = vec[:vector_size]
            else:
                vec = vec + [0.0] * (vector_size - len(vec))
        point_id = engine._generate_chunk_point_id(
            chunk_text=text,
            chunk_index=seg_idx,
            collection_name=collection,
            tenant_id=tenant_id,
        )
        points.append({"id": point_id, "vector": vec, "payload": payload_dict})

    # Phase 4: Single upsert to Qdrant.
    t_upsert = time.monotonic()
    await asyncio.to_thread(engine._ensure_collection_exists, collection)
    await asyncio.to_thread(
        engine._upsert_points_sync, collection_name=collection, points=points
    )
    upsert_ms = (time.monotonic() - t_upsert) * 1000

    total_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "batch-chunk-upsert collection={} items={} segments={} "
        "chunk={:.0f}ms embed={:.0f}ms upsert={:.0f}ms total={:.0f}ms",
        collection, len(payload.items), len(segments),
        chunk_ms, embed_ms, upsert_ms, total_ms,
    )
    return {"chunks_indexed": len(points), "collection": collection}


@app.get("/collections", dependencies=[Depends(require_token)])
async def list_collections(engine: QdrantIndexEngine = Depends(get_engine)) -> dict[str, Any]:  # noqa: B008
    collections = await engine.list_collections()
    return {"collections": collections}


@app.post("/collections", dependencies=[Depends(require_token)])
async def create_collection(
    payload: CollectionCreateRequest,
    engine: QdrantIndexEngine = Depends(get_engine),  # noqa: B008
) -> dict[str, Any]:
    vector_size = payload.vector_size or SETTINGS.index_dims
    distance = payload.distance or SETTINGS.index_metric
    await engine.create_collection(
        collection_name=payload.name,
        vector_size=vector_size,
        distance=distance,
    )
    return {"collection": payload.name, "vector_size": vector_size, "distance": distance}


@app.get("/collections/{name}", dependencies=[Depends(require_token)])
async def get_collection(
    name: str,
    engine: QdrantIndexEngine = Depends(get_engine),  # noqa: B008
) -> dict[str, Any]:
    info = await engine.get_collection_info(collection_name=name)
    return {"collection": name, "info": info}


@app.delete("/collections/{name}", dependencies=[Depends(require_token)])
async def delete_collection(
    name: str,
    engine: QdrantIndexEngine = Depends(get_engine),  # noqa: B008
) -> dict[str, Any]:
    await engine.delete_collection(collection_name=name)
    return {"collection": name, "deleted": True}
