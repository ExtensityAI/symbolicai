"""Tests for the Qdrant RAG API server.

Covers:
- Tenant header enforcement (require_tenant dependency)
- Endpoints that touch Qdrant require X-Tenant-Id
- Endpoints that do not touch Qdrant do not require X-Tenant-Id
- Tenant isolation via mocked engine
- Breaking change: tenant_id in request body returns 422
"""

from __future__ import annotations

import os
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from symai.backend.engines.index.engine_qdrant import QdrantIndexEngine
from symai.backend.settings import SYMAI_CONFIG, SYMSERVER_CONFIG

try:
    from fastapi import HTTPException
    from fastapi.testclient import TestClient
except ImportError:
    HTTPException = None
    TestClient = None

from symai.server.qdrant_rag_api import (
    app,
    get_engine,
    require_tenant,
    require_token,
)


@pytest.fixture
def mock_engine():
    """Return a mocked QdrantIndexEngine."""
    engine = MagicMock()
    engine.upsert = AsyncMock()
    engine.delete = AsyncMock()
    engine.delete_by_filter = AsyncMock()
    engine.count = AsyncMock(return_value=5)
    engine.retrieve = AsyncMock(return_value=[])
    engine.search = AsyncMock(return_value=[])
    engine.chunk_and_upsert = AsyncMock(return_value=3)
    engine._ensure_collection_exists = MagicMock()
    engine._ensure_tenant_index = MagicMock()
    engine._build_query_filter = MagicMock(return_value={"must": []})
    engine.client = MagicMock()
    engine.client.scroll = MagicMock(return_value=([], None))
    engine._normalize_vector = MagicMock(side_effect=lambda v: v if isinstance(v, list) else list(v))
    engine._generate_chunk_point_id = MagicMock(return_value=uuid.uuid4())
    engine._upsert_points_sync = MagicMock()
    engine.chunker = MagicMock()
    engine.chunker_name = "RecursiveChunker"
    engine.chunker.forward = MagicMock(return_value=MagicMock(value=["chunk1", "chunk2"]))
    return engine


@pytest.fixture
def client(mock_engine):
    """Yield a TestClient with auth and engine dependencies overridden."""
    if TestClient is None:
        pytest.skip("fastapi testclient not available")

    async def _noop_token():
        return None

    async def _mock_engine():
        return mock_engine

    app.dependency_overrides[require_token] = _noop_token
    app.dependency_overrides[get_engine] = _mock_engine

    # Ensure SETTINGS.rag_api_token is unset so require_token is a no-op
    with patch("symai.server.qdrant_rag_api.SETTINGS.rag_api_token", None), TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# require_tenant dependency
# ---------------------------------------------------------------------------

class TestRequireTenant:
    """Unit tests for the require_tenant dependency."""

    @pytest.mark.asyncio
    async def test_missing_header_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            await require_tenant(x_tenant_id=None)
        assert exc_info.value.status_code == 400
        assert "Missing or empty X-Tenant-Id header" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_empty_header_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            await require_tenant(x_tenant_id="")
        assert exc_info.value.status_code == 400
        assert "Missing or empty X-Tenant-Id header" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_whitespace_only_header_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            await require_tenant(x_tenant_id="   ")
        assert exc_info.value.status_code == 400
        assert "Missing or empty X-Tenant-Id header" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_valid_header_returns_value(self):
        result = await require_tenant(x_tenant_id="tenant_42")
        assert result == "tenant_42"


# ---------------------------------------------------------------------------
# Endpoints that MUST require X-Tenant-Id
# ---------------------------------------------------------------------------

class TestTenantRequiredEndpoints:
    """Verify Qdrant-touching endpoints reject missing/empty X-Tenant-Id."""

    TENANT_HEADER = {"X-Tenant-Id": "t1"}  # noqa: RUF012

    def test_post_points_missing_tenant(self, client):
        resp = client.post("/points", json={"points": [{"id": 1, "vector": [0.1]}]})
        assert resp.status_code == 400
        assert "Missing or empty X-Tenant-Id header" in resp.json()["detail"]


    def test_post_points_with_tenant(self, client, mock_engine):
        resp = client.post(
            "/points",
            json={"points": [{"id": 1, "vector": [0.1]}]},
            headers=self.TENANT_HEADER,
        )
        assert resp.status_code == 200
        mock_engine.upsert.assert_awaited_once()
        _, kwargs = mock_engine.upsert.call_args
        assert kwargs["tenant_id"] == "t1"

    def test_delete_points_missing_tenant(self, client):
        resp = client.request("DELETE", "/points", json={"ids": [1]})
        assert resp.status_code == 400
        assert "Missing or empty X-Tenant-Id header" in resp.json()["detail"]


    def test_delete_points_with_tenant(self, client, mock_engine):
        resp = client.request("DELETE", "/points", json={"ids": [1]}, headers=self.TENANT_HEADER)
        assert resp.status_code == 200
        _, kwargs = mock_engine.delete.call_args
        assert kwargs["tenant_id"] == "t1"

    def test_count_by_metadata_missing_tenant(self, client):
        resp = client.post("/points/count-by-metadata", json={"metadata": {"k": "v"}})
        assert resp.status_code == 400
        assert "Missing or empty X-Tenant-Id header" in resp.json()["detail"]


    def test_count_by_metadata_with_tenant(self, client, mock_engine):
        resp = client.post(
            "/points/count-by-metadata",
            json={"metadata": {"k": "v"}},
            headers=self.TENANT_HEADER,
        )
        assert resp.status_code == 200
        _, kwargs = mock_engine.count.call_args
        assert kwargs["tenant_id"] == "t1"

    def test_delete_by_metadata_missing_tenant(self, client):
        resp = client.post("/points/delete-by-metadata", json={"metadata": {"k": "v"}})
        assert resp.status_code == 400
        assert "Missing or empty X-Tenant-Id header" in resp.json()["detail"]


    def test_delete_by_metadata_with_tenant(self, client, mock_engine):
        resp = client.post(
            "/points/delete-by-metadata",
            json={"metadata": {"k": "v"}},
            headers=self.TENANT_HEADER,
        )
        assert resp.status_code == 200
        _, kwargs = mock_engine.delete_by_filter.call_args
        assert kwargs["tenant_id"] == "t1"

    def test_list_by_metadata_missing_tenant(self, client):
        resp = client.post("/points/list-by-metadata", json={"metadata": {"k": "v"}})
        assert resp.status_code == 400
        assert "Missing or empty X-Tenant-Id header" in resp.json()["detail"]


    def test_list_by_metadata_with_tenant(self, client, mock_engine):
        resp = client.post(
            "/points/list-by-metadata",
            json={"metadata": {"k": "v"}},
            headers=self.TENANT_HEADER,
        )
        assert resp.status_code == 200
        mock_engine._build_query_filter.assert_called_once()
        _, kwargs = mock_engine._build_query_filter.call_args
        assert kwargs["tenant_id"] == "t1"

    def test_retrieve_missing_tenant(self, client):
        resp = client.post("/retrieve", json={"ids": [1]})
        assert resp.status_code == 400
        assert "Missing or empty X-Tenant-Id header" in resp.json()["detail"]


    def test_retrieve_with_tenant(self, client, mock_engine):
        resp = client.post("/retrieve", json={"ids": [1]}, headers=self.TENANT_HEADER)
        assert resp.status_code == 200
        _, kwargs = mock_engine.retrieve.call_args
        assert kwargs["tenant_id"] == "t1"

    def test_search_missing_tenant(self, client):
        resp = client.post("/search", json={"query": "hello"})
        assert resp.status_code == 400
        assert "Missing or empty X-Tenant-Id header" in resp.json()["detail"]


    def test_search_with_tenant(self, client, mock_engine):
        resp = client.post("/search", json={"query": "hello"}, headers=self.TENANT_HEADER)
        assert resp.status_code == 200
        _, kwargs = mock_engine.search.call_args
        assert kwargs["tenant_id"] == "t1"

    def test_chunk_upsert_missing_tenant(self, client):
        resp = client.post("/chunk-upsert", json={"text": "hello world"})
        assert resp.status_code == 400
        assert "Missing or empty X-Tenant-Id header" in resp.json()["detail"]


    def test_chunk_upsert_with_tenant(self, client, mock_engine):
        resp = client.post(
            "/chunk-upsert",
            json={"text": "hello world"},
            headers=self.TENANT_HEADER,
        )
        assert resp.status_code == 200
        _, kwargs = mock_engine.chunk_and_upsert.call_args
        assert kwargs["tenant_id"] == "t1"

    def test_batch_chunk_upsert_missing_tenant(self, client):
        resp = client.post("/batch-chunk-upsert", json={"items": [{"text": "hello"}]})
        assert resp.status_code == 400
        assert "Missing or empty X-Tenant-Id header" in resp.json()["detail"]


    def test_batch_chunk_upsert_with_tenant(self, client, mock_engine):
        resp = client.post(
            "/batch-chunk-upsert",
            json={"items": [{"text": "hello"}]},
            headers=self.TENANT_HEADER,
        )
        assert resp.status_code == 200
        mock_engine._ensure_tenant_index.assert_called_once()

    def test_chunk_upsert_embed_batch_size(self, client, mock_engine):
        resp = client.post(
            "/chunk-upsert",
            json={"text": "hello world", "embed_batch_size": 42},
            headers=self.TENANT_HEADER,
        )
        assert resp.status_code == 200
        _, kwargs = mock_engine.chunk_and_upsert.call_args
        assert kwargs["embed_batch_size"] == 42

    def test_batch_chunk_upsert_embed_batch_size(self, client):
        with patch("symai.server.qdrant_rag_api.SymaiSymbol") as mock_sym:
            mock_instance = mock_sym.return_value
            mock_instance.embed.return_value.value = [[0.0] * 1536] * 2

            resp = client.post(
                "/batch-chunk-upsert",
                json={
                    "items": [
                        {"text": "one", "skip_chunking": True},
                        {"text": "two", "skip_chunking": True},
                        {"text": "three", "skip_chunking": True},
                        {"text": "four", "skip_chunking": True},
                    ],
                    "embed_batch_size": 2,
                },
                headers=self.TENANT_HEADER,
            )
            assert resp.status_code == 200
            assert resp.json()["chunks_indexed"] == 4

            calls = mock_sym.call_args_list
            assert len(calls) == 2
            assert len(calls[0][0][0]) == 2
            assert len(calls[1][0][0]) == 2
            assert calls[0][0][0] == ["one", "two"]
            assert calls[1][0][0] == ["three", "four"]


# ---------------------------------------------------------------------------
# Endpoints that do NOT require X-Tenant-Id
# ---------------------------------------------------------------------------

class TestNonTenantEndpoints:
    """Verify collection-management and health endpoints accept requests
    without X-Tenant-Id."""

    def test_root_without_tenant(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "SymbolicAI Qdrant RAG" in resp.json()["service"]

    def test_healthz_without_tenant(self, client, mock_engine):
        mock_engine.list_collections = AsyncMock(return_value=["col1"])
        mock_engine.get_collection_info = AsyncMock(return_value={
            "points_count": 0,
            "vectors_count": 0,
            "indexed_vectors_count": 0,
        })
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_list_collections_without_tenant(self, client, mock_engine):
        mock_engine.list_collections = AsyncMock(return_value=["c1", "c2"])
        resp = client.get("/collections")
        assert resp.status_code == 200

    def test_create_collection_without_tenant(self, client, mock_engine):
        mock_engine.create_collection = AsyncMock()
        resp = client.post("/collections", json={"name": "new_col"})
        assert resp.status_code == 200

    def test_get_collection_without_tenant(self, client, mock_engine):
        mock_engine.get_collection_info = AsyncMock(return_value={"name": "new_col"})
        resp = client.get("/collections/new_col")
        assert resp.status_code == 200

    def test_delete_collection_without_tenant(self, client, mock_engine):
        mock_engine.delete_collection = AsyncMock()
        resp = client.delete("/collections/new_col")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Body tenant_id rejection (tenant_id must come from header exclusively)
# ---------------------------------------------------------------------------

class TestBodyTenantIdRejected:
    """Verify that tenant_id in the request body is rejected with 422."""

    TENANT_HEADER = {"X-Tenant-Id": "t1"}  # noqa: RUF012

    def test_points_upsert_body_tenant_id_rejected(self, client):
        resp = client.post(
            "/points",
            json={"points": [{"id": 1, "vector": [0.1]}], "tenant_id": "t2"},
            headers=self.TENANT_HEADER,
        )
        assert resp.status_code == 422
        assert "tenant_id" in str(resp.json()).lower()

    def test_search_body_tenant_id_rejected(self, client):
        resp = client.post(
            "/search",
            json={"query": "hello", "tenant_id": "t2"},
            headers=self.TENANT_HEADER,
        )
        assert resp.status_code == 422

    def test_chunk_upsert_body_tenant_id_rejected(self, client):
        resp = client.post(
            "/chunk-upsert",
            json={"text": "hello", "tenant_id": "t2"},
            headers=self.TENANT_HEADER,
        )
        assert resp.status_code == 422

    def test_batch_chunk_upsert_body_tenant_id_rejected(self, client):
        resp = client.post(
            "/batch-chunk-upsert",
            json={"items": [{"text": "hello"}], "tenant_id": "t2"},
            headers=self.TENANT_HEADER,
        )
        assert resp.status_code == 422

    def test_retrieve_body_tenant_id_rejected(self, client):
        resp = client.post(
            "/retrieve",
            json={"ids": [1], "tenant_id": "t2"},
            headers=self.TENANT_HEADER,
        )
        assert resp.status_code == 422

    def test_delete_points_body_tenant_id_rejected(self, client):
        resp = client.request(
            "DELETE",
            "/points",
            json={"ids": [1], "tenant_id": "t2"},
            headers=self.TENANT_HEADER,
        )
        assert resp.status_code == 422

    def test_metadata_filter_body_tenant_id_rejected(self, client):
        resp = client.post(
            "/points/count-by-metadata",
            json={"metadata": {"k": "v"}, "tenant_id": "t2"},
            headers=self.TENANT_HEADER,
        )
        assert resp.status_code == 422

    def test_metadata_list_body_tenant_id_rejected(self, client):
        resp = client.post(
            "/points/list-by-metadata",
            json={"metadata": {"k": "v"}, "tenant_id": "t2"},
            headers=self.TENANT_HEADER,
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Integration-style tests (require real Qdrant)
# ---------------------------------------------------------------------------

class TestTenantIsolationIntegration:
    """Run against a real Qdrant engine when available."""

    @pytest.fixture
    async def real_engine(self):
        url = (
            SYMSERVER_CONFIG.get("url")
            or SYMAI_CONFIG.get("INDEXING_ENGINE_URL")
            or "http://localhost:6333"
        )
        api_key = SYMAI_CONFIG.get("INDEXING_ENGINE_API_KEY")
        try:
            engine = QdrantIndexEngine(
                url=url,
                api_key=api_key,
                index_name="test_tenant_api",
                index_dims=1536,
            )
            await engine.list_collections()
            return engine
        except Exception:
            pytest.skip("Qdrant server not available")

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        os.getenv("SKIP_INTEGRATION", "0") == "1",
        reason="Integration tests skipped via SKIP_INTEGRATION env var",
    )
    async def test_chunk_upsert_indexes_with_tenant_header(self, real_engine):
        """Ensure chunk-upsert with X-Tenant-Id injects tenant_id into payload."""
        collection = f"test_tenant_api_{int(time.time() * 1000)}"
        await real_engine.create_collection(collection, vector_size=1536)

        # Use the engine directly to verify the parameter plumbing works
        chunks = await real_engine.chunk_and_upsert(
            collection_name=collection,
            text="Hello world from integration test.",
            tenant_id="integration_tenant",
        )
        assert chunks > 0

        # Retrieve and verify tenant_id is in payload
        results = await real_engine.search(
            collection, [0.1] * 1536, limit=10, tenant_id="integration_tenant"
        )
        assert len(results) > 0
        for r in results:
            payload = getattr(r, "payload", None) or {}
            assert payload.get("tenant_id") == "integration_tenant"

        await real_engine.delete_collection(collection)
