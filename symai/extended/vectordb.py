import os
import gzip
import pickle

import numpy as np

from ..symbol import Symbol, Expression
from ..interfaces import Interface
from .metrics import (
    adams_similarity,
    cosine_similarity,
    derridaean_similarity,
    dot_product,
    euclidean_metric,
    ranking_algorithm_sort
)


MAX_BATCH_SIZE = 2048 # Maximum batch size for embedding function


class VectorDB(Expression):
    """
    A simple vector database that stores vectors and documents in memory.

    Inspired by the foundational research of John Dagdelen from the University of California, Berkeley and
    his unique approach to problem solving with vector databases: https://github.com/jdagdelen/hyperDB

    Parameters
    ----------
    documents : list, optional
        A list of documents to add to the database.
    vectors : list, optional
        A list of vectors to add to the database.
    embedding_model : callable, optional
        The reference to a callable embedding model used in the default `embedding_function`; defaults to `ExtensityAI/embeddings` which is `all-mpnet-base-v2` based.
    embedding_function : callable, optional
        A function that takes in a list of documents and returns a list of vectors; defaults to `_get_embedding` which uses the callable `embedding_model`.
    similarity_metric : str, optional
        The similarity metric to use when querying the database. Can be either 'dot', 'cosine', 'euclidean', 'adams', or 'derrida'.
    batch_size : int, optional
        The batch size to use when embedding documents into vectors.
    load_on_init : bool, optional
        Whether or not to load the database from a file on initialization.
    index_dims : int, optional
        The number of dimensions to use when creating a new index. Defaults to 768.
    top_k : int, optional
        The number of results to return when querying the database. Defaults to 5.

    Attributes
    ----------
    documents : list
        A list of documents in the database.
    vectors : numpy.ndarray
        A numpy array of vectors in the database.
    embedding_function : function
        The function used to embed documents into vectors.
    similarity_metric : function
        The similarity metric used to query the database.

    Methods
    -------
    dict(vectors=False)
        Returns a list of documents in the database.
    add(documents, vectors=None)
        Adds a document or list of documents to the database.
    add_document(document, vector=None)
        Adds a document to the database.
    remove_document(index)
        Removes a document from the database.
    add_documents(documents, vectors=None)
        Adds a list of documents to the database.
    save(storage_file)
        Saves the database to a file.
    load(storage_file)
        Loads the database from a file.
    forward(query_text, top_k=5, return_similarities=True)
        Queries the database for similar documents.

    Examples
    --------
    >>> from symai.extended import VectorDB
    >>> db = VectorDB()
    >>> db.add("Hello, World!")
    >>> db("Hello, World!")
    """
    _default_documents          = []
    _default_vectors            = None
    _default_embedding_model    = None
    _default_batch_size         = MAX_BATCH_SIZE
    _default_similarity_metric  = "cosine"
    _default_embedding_function = None
    _default_index_dims         = 768
    _default_top_k              = 5
    _default_storage_path       = os.path.join(os.path.expanduser("~"), ".symai", "localdb")
    _default_index_name         = "dataindex"

    def __init__(
        self,
        documents=_default_documents,
        vectors=_default_vectors,
        embedding_model=_default_embedding_model,
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
        #
        self.documents          = []
        self.vectors            = None
        # init basic properties
        self.batch_size         = batch_size
        self.index_dims         = index_dims
        self.index_top_k        = top_k
        self.index_name         = index_name
        # init embedding function
        self.model              = embedding_model or Interface('ExtensityAI/embeddings')
        self.embedding_function = embedding_function or self._get_embedding
        if vectors is not None:
            self.vectors        = vectors
            self.documents      = documents
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
            raise Exception("Similarity metric not supported. Please use either 'dot', 'cosine', 'euclidean', 'adams', or 'derrida'.")

        if load_on_init:
            # If load_on_init is a string, use it as the storage file
            if isinstance(load_on_init, str):
                path = os.path.join(load_on_init, f"{self.index_name}.pkl")
                self.load(path)
            else:
                self.load()

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
        # unwrap the documents if they are a Symbol
        if isinstance(documents, Symbol):
            documents = documents.value
        # if the documents are a list of Symbols, unwrap them
        if len(documents) == 0:
            return []

        if isinstance(documents, list):
            # If the documents are a list of dictionaries, extract the text from the dictionary
            if isinstance(documents[0], dict):
                texts = []
                # If a key is specified, extract the text from the dictionary using the key
                if isinstance(key, str):
                    if "." in key:
                        key_chain = key.split(".")
                    else:
                        key_chain = [key]
                    for doc in documents:
                        for key in key_chain:
                            doc   = doc[key]
                        texts.append(doc.replace("\n", " "))
                # If no key is specified, extract the text from the dictionary using all keys
                elif key is None:
                    for doc in documents:
                        text      = ", ".join([f"{key}: {value}" for key, value in doc.items()])
                        texts.append(text)
            # If the documents are a list of strings, use the strings as the documents
            elif isinstance(documents[0], str):
                texts = documents
            # If the documents are a list of lists, use the lists as the documents
        batches    = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        embeddings = []
        # Embed the documents in batches
        for batch in batches:
            # Extend the embeddings list with the embeddings from the batch
            emb = self.model(batch)
            if len(emb.shape) == 1:
                embeddings.append(emb)
            elif len(emb.shape) == 2:
                for i in range(emb.shape[0]):
                    embeddings.append(emb[i])
            else:
                raise Exception("Embeddings must be a 1D or 2D array.")
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
                    zip(self.documents, self.vectors)
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

    def add_document(self, document: dict, vector=None):
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
            raise ValueError("All vectors must have the same length.")
        # convert the vector to a numpy array if it is not already
        if type(vector) == list:
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
        for vector, document in zip(vectors, documents):
            self.add_document(document, vector)

    def clear(self):
        """
        Clears the database.

        """
        self.vectors   = None
        self.documents = []

    def save(self, storage_file: str = None):
        """
        Saves the database to a file.

        Parameters
        ----------
        storage_file : str, optional
            The file to save the database to.

        """
        if storage_file is None:
            # use path to home directory by default
            storage_path = os.path.join(os.path.expanduser("~"), ".symai", "localdb")
            os.makedirs(storage_path, exist_ok=True)
            storage_file = os.path.join(storage_path, f"{self.index_name}.pkl")

        data = {"vectors": self.vectors, "documents": self.documents}
        if storage_file.endswith(".gz"):
            with gzip.open(storage_file, "wb") as f:
                pickle.dump(data, f)
        else:
            with open(storage_file, "wb") as f:
                pickle.dump(data, f)

    def load(self, storage_file : str = None):
        """
        Loads the database from a file.

        Parameters
        ----------
        storage_file : str, optional
            The file to load the database from.

        """
        if storage_file is None:
            # use path to home directory by default
            storage_path = os.path.join(os.path.expanduser("~"), ".symai", "localdb")
            # create dir on first load if never used
            os.makedirs(storage_path, exist_ok=True)
            storage_file = os.path.join(storage_path, f"{self.index_name}.pkl")

        # return since nothing to load
        if not os.path.exists(storage_file):
            return

        if storage_file.endswith(".gz"):
            with gzip.open(storage_file, "rb") as f:
                data   = pickle.load(f)
        else:
            with open(storage_file, "rb") as f:
                data   = pickle.load(f)
        self.vectors   = data["vectors"].astype(np.float32)
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
        # use path to home directory by default
        storage_path = os.path.join(os.path.expanduser("~"), ".symai", "localdb")
        # create dir on first load if never used
        os.makedirs(storage_path, exist_ok=True)
        storage_file = os.path.join(storage_path, f"{index_name}.pkl")

        # return since nothing to load
        if not os.path.exists(storage_file):
            return

        # remove the file
        os.remove(storage_file)

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
        assert self.vectors is not None, f"Error: Cannot query the database without prior insertion / initialization."
        top_k          = top_k or self.index_top_k
        query_vector   = self.embedding_function([query])[0] if vector is None else vector
        if type(query_vector) == list:
            query_vector = np.array(query_vector)
        ranked_results, similarities = ranking_algorithm_sort(
            self.vectors, query_vector, top_k=top_k, metric=self.similarity_metric
        )
        if return_similarities:
            return list(zip([self.documents[index] for index in ranked_results], similarities))
        return [self.documents[index] for index in ranked_results]
