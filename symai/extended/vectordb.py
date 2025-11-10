import gzip
import logging
import pickle
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..backend.settings import HOME_PATH, SYMAI_CONFIG
from ..interfaces import Interface
from ..symbol import Expression, Symbol
from ..utils import UserMessage
from .metrics import (
    adams_similarity,
    cosine_similarity,
    derridaean_similarity,
    dot_product,
    euclidean_metric,
    ranking_algorithm_sort,
)

logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('datasets').setLevel(logging.WARNING)


class VectorDB(Expression):
    _default_documents: ClassVar[list] = []
    _default_vectors: ClassVar[np.ndarray | None] = None
    _default_batch_size: ClassVar[int] = 2048
    _default_similarity_metric: ClassVar[str] = "cosine"
    _default_embedding_function: ClassVar[object | None] = None
    _default_index_dims: ClassVar[int] = 768
    _default_top_k: ClassVar[int] = 5
    _default_storage_path: ClassVar[Path] = HOME_PATH / "localdb"
    _default_index_name: ClassVar[str] = "dataindex"
    def __init__(
        self,
        documents=_default_documents,
        vectors=_default_vectors,
        embedding_function=_default_embedding_function,
        similarity_metric=_default_similarity_metric,
        batch_size=_default_batch_size,
        load_on_init=_default_storage_path,
        index_dims=_default_index_dims,
        top_k=_default_top_k,
        index_name=_default_index_name,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = deepcopy(SYMAI_CONFIG)
        self.documents = []
        self.vectors = None
        # init basic properties
        self.batch_size = batch_size
        self.index_dims = index_dims
        self.index_top_k = top_k
        self.index_name = index_name
        # init embedding function
        self._init_embedding_model()
        self.embedding_function = embedding_function or self._get_embedding
        if vectors is not None:
            self.vectors = vectors
            self.documents = documents
        else:
            self.add_documents(documents)

        if "dot" in similarity_metric:
            self.similarity_metric = dot_product
        elif "cosine" in similarity_metric:
            self.similarity_metric = cosine_similarity
        elif "euclidean" in similarity_metric:
            self.similarity_metric = euclidean_metric
        elif "derrida" in similarity_metric:
            self.similarity_metric = derridaean_similarity
        elif "adams" in similarity_metric:
            self.similarity_metric = adams_similarity
        else:
            UserMessage("Similarity metric not supported. Please use either 'dot', 'cosine', 'euclidean', 'adams', or 'derrida'.", raise_with=ValueError)

        if load_on_init:
            if isinstance(load_on_init, (str, Path)):
                path = Path(load_on_init) / f"{self.index_name}.pkl"
                self.load(path)
            else:
                self.load()

    def _init_embedding_model(self):
        if self.config['EMBEDDING_ENGINE_API_KEY'] is None or  self.config['EMBEDDING_ENGINE_API_KEY'] == '':
            self.model = Interface('ExtensityAI/embeddings') # default to local model
        else:
            self.model = lambda x: Symbol(x).embedding

    def _unwrap_documents(self, documents):
        if isinstance(documents, Symbol):
            return documents.value
        return documents

    def _to_texts(self, documents, key):
        if not isinstance(documents, list):
            self._raise_texts_unassigned()
        if len(documents) == 0:
            return []
        first_document = documents[0]
        if isinstance(first_document, dict):
            return self._texts_from_dicts(documents, key)
        if isinstance(first_document, str):
            return documents
        return self._raise_texts_unassigned()

    def _texts_from_dicts(self, documents, key):
        if isinstance(key, str):
            key_chain = key.split(".") if "." in key else [key]
            return [self._resolve_key_chain(doc, key_chain).replace("\n", " ") for doc in documents]
        if key is None:
            return [
                ", ".join([f"{dict_key}: {value}" for dict_key, value in doc.items()])
                for doc in documents
            ]
        return self._raise_texts_unassigned()

    def _resolve_key_chain(self, document, key_chain):
        current_document = document
        for chain_key in key_chain:
            current_document = current_document[chain_key]
        return current_document

    def _embed_batch(self, batch):
        emb = self.model(batch)
        if len(emb.shape) == 1:
            return [emb]
        if len(emb.shape) == 2:
            return [emb[index] for index in range(emb.shape[0])]
        return UserMessage("Embeddings must be a 1D or 2D array.", raise_with=ValueError)

    def _raise_texts_unassigned(self):
        error_message = "local variable 'texts' referenced before assignment"
        raise UnboundLocalError(error_message)

    def _get_embedding(self, documents, key=None):
        """
        Get embeddings from a list of documents.

        Parameters
        ----------
        documents : list
            A list of documents to embed.
        key : str, optional
            The key to use when extracting text from a dictionary.

        Returns
        -------
        embeddings : numpy.ndarray
            A numpy array of embeddings.
        """
        documents = self._unwrap_documents(documents)
        if len(documents) == 0:
            return []
        texts = self._to_texts(documents, key)
        batches = [texts[index : index + self.batch_size] for index in range(0, len(texts), self.batch_size)]
        embeddings = []
        for batch in batches:
            embeddings.extend(self._embed_batch(batch))
        return embeddings

    def dict(self, vectors=False):
        """
        Returns a list of documents in the database.

        Parameters
        ----------
        vectors : bool, optional
            Whether or not to include vectors in the returned list.

        Returns
        -------
        documents : list
            A list of documents in the database.
        """
        if vectors:
            return [
                {"document": document, "vector": vector.tolist(), "index": index}
                for index, (document, vector) in enumerate(
                    zip(self.documents, self.vectors, strict=False)
                )
            ]
        return [
            {"document": document, "index": index}
            for index, document in enumerate(self.documents)
        ]

    def add(self, documents, vectors=None):
        """
        Adds a document or list of documents to the database.

        Parameters
        ----------
        documents : dict or list
            A document or list of documents to add to the database.
        vectors : numpy.ndarray, optional
            A vector or list of vectors to add to the database.

        """
        # unwrap the documents if they are a Symbol
        if isinstance(documents, Symbol):
            documents = documents.value
        if not isinstance(documents, list):
            return self.add_document(documents, vectors)
        self.add_documents(documents, vectors)
        return None

    def add_document(self, document: Mapping[str, Any], vector=None):
        """
        Adds a document to the database.

        Parameters
        ----------
        document : dict
            A document to add to the database.
        vector : numpy.ndarray, optional
            A vector to add to the database.

        """
        vector = (vector if vector is not None else self.embedding_function([document])[0])
        if self.vectors is None:
            self.vectors = np.empty((0, len(vector)), dtype=np.float32)
        elif len(vector) != self.vectors.shape[1]:
            UserMessage("All vectors must have the same length.", raise_with=ValueError)
        # convert the vector to a numpy array if it is not already
        if isinstance(vector, list):
            vector = np.array(vector)
        self.vectors = np.vstack([self.vectors, vector]).astype(np.float32)
        self.documents.append(document)

    def remove_document(self, index):
        """
        Removes a document from the database.

        Parameters
        ----------
        index : int
            The index of the document to remove.

        """
        self.vectors = np.delete(self.vectors, index, axis=0)
        self.documents.pop(index)

    def add_documents(self, documents, vectors=None):
        """
        Adds a list of documents to the database.

        Parameters
        ----------
        documents : list
            A list of documents to add to the database.
        vectors : numpy.ndarray, optional
            A list of vectors to add to the database.

        """
        if not documents:
            return
        vectors = vectors or np.array(self.embedding_function(documents)).astype(np.float32)
        for vector, document in zip(vectors, documents, strict=False):
            self.add_document(document, vector)

    def clear(self):
        """
        Clears the database.

        """
        self.vectors   = None
        self.documents = []

    def save(self, storage_file: str | None = None):
        """
        Saves the database to a file.

        Parameters
        ----------
        storage_file : str, optional
            The file to save the database to.

        """
        if storage_file is None:
            storage_file = HOME_PATH / "localdb" / f"{self.index_name}.pkl"
            storage_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            storage_file = Path(storage_file)

        data = {"vectors": self.vectors, "documents": self.documents}
        if storage_file.suffix == ".gz":
            with gzip.open(storage_file, "wb") as f:
                pickle.dump(data, f)
        else:
            with storage_file.open("wb") as f:
                pickle.dump(data, f)

    def load(self, storage_file : str | None = None):
        """
        Loads the database from a file.

        Parameters
        ----------
        storage_file : str, optional
            The file to load the database from.

        """
        if storage_file is None:
            storage_file = HOME_PATH / "localdb" / f"{self.index_name}.pkl"
            storage_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            storage_file = Path(storage_file)

        # return since nothing to load
        if not storage_file.exists():
            return

        if storage_file.suffix == ".gz":
            with gzip.open(storage_file, "rb") as f:
                data = pickle.load(f)
        else:
            with storage_file.open("rb") as f:
                data = pickle.load(f)

        self.vectors = data["vectors"].astype(np.float32) if data["vectors"] is not None else None
        self.documents = data["documents"]

    def purge(self, index_name : str):
        """
        Purges the database file from your machine, but does not delete the database from memory.
        Use the `clear` method to clear the database from memory.
        ATTENTION! This is a permanent action and cannot be undone.

        Parameters
        ----------
        index_name : str
            The index file to purge the database from your system.

        """
        index_name = index_name or self.index_name
        assert index_name, "Error: Please provide an index name to purge the database."
        # symai folder
        symai_folder = Path(HOME_PATH)
        # use path to home directory by default
        storage_path = symai_folder / "localdb"
        # create dir on first load if never used
        storage_path.mkdir(parents=True, exist_ok=True)
        storage_file = storage_path / f"{index_name}.pkl"
        if storage_file.exists():
            # remove the file
            storage_file.unlink()
        self.clear()

    def forward(self, query=None, vector=None, top_k=None, return_similarities=True):
        """
        Queries the database for similar documents.

        Parameters
        ----------
        query : str or dict
            The query to search for.
        top_k : int, optional
            The number of results to return.
        return_similarities : bool, optional
            Whether or not to return the similarity scores.

        Returns
        -------
        results : list
            A list of results.

        """
        assert self.vectors is not None, "Error: Cannot query the database without prior insertion / initialization."
        top_k = top_k or self.index_top_k
        query_vector = self.embedding_function([query])[0] if vector is None else vector
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector)
        ranked_results, similarities = ranking_algorithm_sort(
            self.vectors, query_vector, top_k=top_k, metric=self.similarity_metric
        )
        if return_similarities:
            return list(zip([self.documents[index] for index in ranked_results], similarities, strict=False))
        return [self.documents[index] for index in ranked_results]
