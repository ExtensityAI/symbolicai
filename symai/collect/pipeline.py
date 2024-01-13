import json
import logging

from bson.objectid import ObjectId
from datetime import datetime
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.mongo_client import MongoClient
from typing import Any, Dict, List, Optional

from ..backend.settings import SYMAI_CONFIG


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
    elif isinstance(obj, dict):
        # For dictionaries, serialize each value. Keep keys as strings.
        return {str(key): rec_serialize(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        # For lists, tuples, and sets, serialize each element.
        return [rec_serialize(elem) for elem in obj]
    else:
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
        self.client: Optional[MongoClient]      = None
        self.db: Optional[Database]             = None
        self.collection: Optional[Collection]   = None

    def __enter__(self) -> 'CollectionRepository':
        self.connect()
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> None:
        self.close()

    def ping(self) -> bool:
        if not self.support_community:
            return False
        # Send a ping to confirm a successful connection
        try:
            self.client.admin.command('ping')
            return True
        except Exception as e:
            print("Connection failed: " + str(e))
            return False

    def add(self, forward: Any, engine: Any, metadata: Dict[str, Any] = {}) -> Any:
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
        except Exception as e:
            logger.debug(f"Error adding record: {e}")
            return None

    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        if not self.support_community:
            return None
        return self.collection.find_one({'_id': ObjectId(record_id)}) if self.collection else None

    def update(self,
               record_id: str,
               forward: Optional[Any]             = None,
               engine: Optional[str]              = None,
               metadata: Optional[Dict[str, Any]] = None) -> Any:
        if not self.support_community:
            return None
        updates: Dict[str, Any] = {'updated_at': datetime.now()}
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

    def list(self, filters: Optional[Dict[str, Any]] = None, limit: int = 0) -> List[Dict[str, Any]]:
        if not self.support_community:
            return None
        if filters is None:
            filters = {}
        return list(self.collection.find(filters).limit(limit)) if self.collection else []

    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
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
            print("[WARN] MongoClient: Connection failed: " + str(e))

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
