# vector_store/store.py

"""
Vector database storage module for managing document embeddings.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

# Setup module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class BaseVectorStore:
    """Base class for vector storage systems"""

    def __init__(self):
        """Initialize the base vector store"""
        pass

    def add_documents(self, texts: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]],
                      ids: List[str]) -> None:
        """
        Add documents to the vector store.

        Args:
            texts: List of document texts
            embeddings: Matrix of document embeddings
            metadatas: List of document metadata
            ids: List of document IDs
        """
        raise NotImplementedError("Subclasses must implement this method")

    def query(self, query_embedding: np.ndarray, n_results: int = 5) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.

        Args:
            query_embedding: Embedding vector for the query
            n_results: Number of results to return

        Returns:
            Query results
        """
        raise NotImplementedError("Subclasses must implement this method")


class ChromaVectorStore(BaseVectorStore):
    """Vector storage using ChromaDB"""

    def __init__(
            self,
            collection_name: str = "document_embeddings",
            persist_dir: str = "./chroma_db"
    ):
        """
        Initialize the ChromaDB vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_dir: Directory where ChromaDB will be saved
        """
        super().__init__()

        self.collection_name = collection_name
        self.persist_dir = persist_dir

        # Initialize ChromaDB client
        try:
            import chromadb
            self.chroma_client = chromadb.PersistentClient(path=persist_dir)
            logger.info(f"ChromaDB client initialized with persistence directory: {persist_dir}")
        except ImportError:
            logger.error("Failed to import ChromaDB. Make sure it's installed using 'pip install chromadb'")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            raise

        self.collection = None

    def get_or_create_collection(self, force_recreate: bool = False) -> Any:
        """
        Get an existing collection or create a new one.

        Args:
            force_recreate: If True, recreate the collection even if it exists

        Returns:
            The ChromaDB collection
        """
        # Check if we need to recreate the collection
        if force_recreate and self.collection_name in [col.name for col in self.chroma_client.list_collections()]:
            logger.info(f"Force recreate flag set. Deleting existing collection: {self.collection_name}")
            self.chroma_client.delete_collection(name=self.collection_name)
            self.collection = self.chroma_client.create_collection(name=self.collection_name)
            logger.info(f"Recreated collection: {self.collection_name}")
        else:
            # Try to get existing collection or create a new one
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
                logger.info(
                    f"Found existing collection '{self.collection_name}' with {self.collection.count()} entries")
            except Exception:
                logger.info(f"Collection '{self.collection_name}' not found, creating new one")
                self.collection = self.chroma_client.create_collection(name=self.collection_name)
                logger.info(f"Created new collection: {self.collection_name}")

        return self.collection

    def get_existing_files(self) -> Set[str]:
        """
        Get a set of filenames that already exist in the collection.

        Returns:
            Set of filenames already in the database
        """
        if self.collection is None:
            self.get_or_create_collection()

        existing_files = set()
        if self.collection.count() > 0:
            # Query all existing items to get metadata
            logger.debug("Retrieving existing files from collection metadata")
            result = self.collection.get(include=["metadatas"])
            for metadata in result["metadatas"]:
                if "filename" in metadata:
                    existing_files.add(metadata["filename"])

            logger.debug(f"Found {len(existing_files)} existing files in collection")
        else:
            logger.debug("Collection is empty, no existing files found")

        return existing_files

    def add_documents(self, texts: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]],
                      ids: List[str]) -> None:
        """
        Add documents to the ChromaDB collection.

        Args:
            texts: List of document texts
            embeddings: Matrix of document embeddings
            metadatas: List of document metadata
            ids: List of document IDs
        """
        if self.collection is None:
            self.get_or_create_collection()

        try:
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Successfully added {len(texts)} documents to collection")
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {str(e)}")
            raise

    def query(self, query_embedding: np.ndarray, n_results: int = 5) -> Dict[str, Any]:
        """
        Query the ChromaDB collection for similar documents.

        Args:
            query_embedding: Embedding vector for the query
            n_results: Number of results to return

        Returns:
            Query results from ChromaDB
        """
        if self.collection is None:
            self.get_or_create_collection()

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            logger.info(f"Query returned {len(results['documents'][0])} results")
            return results
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise