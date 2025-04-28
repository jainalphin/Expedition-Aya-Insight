# vector_store/__init__.py

"""
Vector store package for managing document embeddings.
"""

from .store import BaseVectorStore, ChromaVectorStore
from .document_manager import DocumentManager

__all__ = [
    'BaseVectorStore',
    'ChromaVectorStore',
    'DocumentManager'
]