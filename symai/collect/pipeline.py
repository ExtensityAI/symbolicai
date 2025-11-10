from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from bson.objectid import ObjectId
from pymongo.mongo_client import MongoClient

from ..backend.settings import SYMAI_CONFIG
from ..utils import UserMessage

if TYPE_CHECKING:
    from pymongo.collection import Collection
    from pymongo.database import Database
else:
    Collection = Database = Any

logger = logging.getLogger(__name__)


def rec_serialize(obj):
    """
    Recursively serialize a given object into a string representation, handling
    nested structures like lists and dictionaries.

    :param obj: The object to be serialized.
    :return: A string representation of the serialized object.
    """
    if isinstance(obj, (int, float, bool)):
        # For simple types, return the string representation directly.
        return obj
    if isinstance(obj, dict):
        # For dictionaries, serialize each value. Keep keys as strings.
        return {str(key): rec_serialize(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        # For lists, tuples, and sets, serialize each element.
        return [rec_serialize(elem) for elem in obj]
    # Attempt JSON serialization first, then fall back to str(...)
    try:
        return json.dumps(obj)
    except TypeError:
        return str(obj)


class CollectionRepository:
    def __init__(self) -> None:
        self.support_community: bool            = SYMAI_CONFIG["SUPPORT_COMMUNITY"]
        self.uri: str                           = SYMAI_CONFIG["COLLECTION_URI"]
        self.db_name: str                       = SYMAI_CONFIG["COLLECTION_DB"]
        self.collection_name: str               = SYMAI_CONFIG["COLLECTION_STORAGE"]
        self.client: MongoClient | None      = None
        self.db: Database | None             = None
        self.collection: Collection | None   = None

    def __enter__(self) -> CollectionRepository:
        self.connect()
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any | None) -> None:
        self.close()

    def ping(self) -> bool:
        if not self.support_community:
            return False
        # Send a ping to confirm a successful connection
        try:
            self.client.admin.command('ping')
            return True
        except Exception as e:
            UserMessage(f"Connection failed: {e}")
            return False

    def add(self, forward: Any, engine: Any, metadata: dict[str, Any] | None = None) -> Any:
        if metadata is None:
            metadata = {}
        if not self.support_community:
            return None
        record = {
            'forward': forward,
            'engine': engine,
            'metadata': metadata,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
        try: # assure that adding a record does never cause a system error
            return self.collection.insert_one(record).inserted_id if self.collection else None
        except Exception:
            return None

    def get(self, record_id: str) -> dict[str, Any] | None:
        if not self.support_community:
            return None
        return self.collection.find_one({'_id': ObjectId(record_id)}) if self.collection else None

    def update(self,
               record_id: str,
               forward: Any | None             = None,
               engine: str | None              = None,
               metadata: dict[str, Any] | None = None) -> Any:
        if not self.support_community:
            return None
        updates: dict[str, Any] = {'updated_at': datetime.now()}
        if forward is not None:
            updates['forward']  = forward
        if engine is not None:
            updates['engine']   = engine
        if metadata is not None:
            updates['metadata'] = metadata

        return self.collection.update_one({'_id': ObjectId(record_id)}, {'$set': updates}) if self.collection else None

    def delete(self, record_id: str) -> Any:
        if not self.support_community:
            return None
        return self.collection.delete_one({'_id': ObjectId(record_id)}) if self.collection else None

    def list(self, filters: dict[str, Any] | None = None, limit: int = 0) -> list[dict[str, Any]]:
        if not self.support_community:
            return None
        if filters is None:
            filters = {}
        return list(self.collection.find(filters).limit(limit)) if self.collection else []

    def count(self, filters: dict[str, Any] | None = None) -> int:
        if not self.support_community:
            return None
        if filters is None:
            filters = {}
        return self.collection.count_documents(filters) if self.collection else 0

    def connect(self) -> None:
        try:
            if self.client is None and self.support_community:
                self.client = MongoClient(self.uri)
                self.db     = self.client[self.db_name]
                self.collection = self.db[self.collection_name]
        except Exception as e:
            # disable retries
            self.client     = False
            self.db         = None
            self.collection = None
            UserMessage(f"[WARN] MongoClient: Connection failed: {e}")

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
