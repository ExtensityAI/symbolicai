import json
import os

import pytest

from symai.backend.engines.index.engine_vectordb import VectorDBResult
from symai.backend.settings import HOME_PATH, SYMAI_CONFIG
from symai.extended.interfaces.naive_vectordb import naive_vectordb
from symai.utils import CustomUserWarning

if not bool(SYMAI_CONFIG.get("EMBEDDING_ENGINE_API_KEY")):
    CustomUserWarning(
        "Since no EMBEDDING_ENGINE_API_KEY key is set, the embedding engine will try to default to a local embedding model based on SentenceTransformers and 'ExtensityAI/embeddings' plugin. Make sure you use an 'EMBEDDING_ENGINE_MODEL' that is compatible with SentenceTransformers (e.g., 'all-mpnet-base-v2')."
    )

def test_add_and_search_single_document():
    index_name = "testindex_single"
    db = naive_vectordb(index_name=index_name)
    db("Hello world", operation="add")
    result = db("Hello world", operation="search")
    assert isinstance(result, VectorDBResult)
    assert result.value, "Expected at least one match to be returned."
    assert "Hello world" in result.value[0]

def test_add_and_search_multiple_documents():
    index_name = "testindex_multiple"
    docs = ["Alpha document", "Beta entry", "Gamma text"]
    db = naive_vectordb(index_name=index_name)
    db(docs, operation="add")
    result = db("Beta entry", operation="search", top_k=1)
    assert isinstance(result, VectorDBResult)
    assert result.value, "Expected at least one match to be returned."
    assert result.value[0] == "Beta entry"

def test_save_load_purge():
    import time
    index_name = "testindex_save_load_purge"
    storage_dir = HOME_PATH / "localdb"
    storage_file = storage_dir / f"{index_name}.pkl"
    docs = ["Alpha document", "Beta entry", "Gamma text"]

    db = naive_vectordb(index_name=index_name)
    db(docs, operation="add")
    db("save", operation="config")

    assert storage_file.exists(), "Index storage file was not created after save."

    db("purge", operation="config")
    for i in range(5):
        if not storage_file.exists():
            break
        time.sleep(0.1)
    assert not storage_file.exists(), "Index storage file still exists after purge."

    db(docs, operation="add")
    db("save", operation="config")
    assert storage_file.exists(), "Index storage file missing for loading."

    db("load", operation="config", storage_file=storage_dir.as_posix())
    search_result = db("Beta entry", operation="search", top_k=1)
    assert isinstance(search_result, VectorDBResult)
    assert search_result.value, "Search returned no results after load."
    assert search_result.value[0] == "Beta entry"

    db("purge", operation="config")
    assert not storage_file.exists(), "Index file should be removed after final purge."
