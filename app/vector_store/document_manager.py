# vector_store/document_manager.py

"""
Document manager for integrating document processing with vector storage.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

# Setup module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DocumentManager:
    """
    Document manager that connects document processing, embedding generation,
    and vector storage to provide a complete workflow.
    """

    def __init__(
            self,
            vector_store,
            embedding_generator,
            force_recreate: bool = False
    ):
        """
        Initialize the document manager.

        Args:
            vector_store: Vector storage instance
            embedding_generator: Embedding generator instance
            force_recreate: Whether to force recreate the vector store collection
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator

        # Initialize the vector store
        self.vector_store.get_or_create_collection(force_recreate=force_recreate)

    def filter_new_documents(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out documents that already exist in the collection.
        Also filters out documents with empty text.

        Args:
            data: List of dictionaries with 'text' and 'filename' keys

        Returns:
            List of documents that don't already exist and have non-empty text
        """
        existing_files = self.vector_store.get_existing_files()

        logger.info(f"Filtering {len(data)} input documents")

        new_data = []
        empty_text_count = 0
        already_exists_count = 0

        for item in data:
            # Check if text is empty
            if not item.get('text') or item['text'].strip() == "":
                empty_text_count += 1
                logger.warning(f"Empty text found in document with filename: {item.get('filename', 'unknown')}")
                continue

            # Check if file already exists
            if item.get('filename') in existing_files:
                already_exists_count += 1
                logger.debug(f"Document already exists: {item.get('filename')}")
                continue

            new_data.append(item)

        logger.info(
            f"Filtered results: {len(new_data)} new documents, {empty_text_count} empty documents, "
            f"{already_exists_count} existing documents"
        )

        return new_data

    def validate_documents(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate documents to ensure they have required fields and non-empty text.

        Args:
            data: List of documents to validate

        Returns:
            List of valid documents
        """
        valid_docs = []
        invalid_count = 0

        for i, doc in enumerate(data):
            if not isinstance(doc, dict):
                logger.error(f"Document at index {i} is not a dictionary")
                invalid_count += 1
                continue

            if 'text' not in doc:
                logger.error(f"Document at index {i} is missing 'text' field")
                invalid_count += 1
                continue

            if 'filename' not in doc:
                logger.error(f"Document at index {i} is missing 'filename' field")
                invalid_count += 1
                continue

            if not doc['text'] or doc['text'].strip() == "":
                logger.error(f"Document at index {i} with filename '{doc['filename']}' has empty text")
                invalid_count += 1
                continue

            valid_docs.append(doc)

        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} invalid documents out of {len(data)}")

        return valid_docs

    def process_documents(self, data: List[Dict[str, Any]], force_recreate: bool = False) -> None:
        """
        Process a list of documents, generate embeddings, and store them.

        Args:
            data: List of dictionaries with 'text' and 'filename' keys
            force_recreate: If True, recreate the collection

        Returns:
            None
        """
        logger.info(f"Processing {len(data)} documents")

        # Validate documents
        valid_data = self.validate_documents(data)

        if not valid_data:
            logger.error("No valid documents to process")
            return

        # If not forcing recreation, filter new documents
        if not force_recreate:
            new_data = self.filter_new_documents(valid_data)

            if not new_data:
                logger.info("All files are already in the database. No new files to add.")
                return

            logger.info(f"Found {len(new_data)} new documents to process out of {len(valid_data)} total.")
        else:
            # If we're recreating, treat all documents as new
            new_data = valid_data
            logger.info(f"Force recreate: processing all {len(valid_data)} documents as new")

        # Extract texts for embeddings
        texts = [item['text'] for item in new_data]

        # Generate embeddings
        try:
            embeddings = self.embedding_generator.generate_embeddings(texts)
        except Exception as e:
            logger.error(f"Failed to add documents due to embedding error: {str(e)}")
            raise

        # Generate unique IDs for the new documents
        existing_count = self.vector_store.collection.count() if self.vector_store.collection else 0
        new_ids = [f"doc_{existing_count + i}" for i in range(len(new_data))]

        # Add documents to vector store
        self.vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=[{"filename": item['filename']} for item in new_data],
            ids=new_ids
        )

    def query_similar_documents(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Query for documents similar to the query text.

        Args:
            query_text: Text to search for
            n_results: Number of results to return

        Returns:
            Query results
        """
        if not query_text or query_text.strip() == "":
            error_msg = "Cannot query with empty text"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Querying with text of length {len(query_text)}")

        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embeddings([query_text])[0]

        # Query the vector store
        return self.vector_store.query(query_embedding, n_results=n_results)