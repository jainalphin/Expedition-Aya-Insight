# embedding/__init__.py

"""
Embedding generation package for creating vector embeddings from text.
"""

from .generator import BaseEmbeddingGenerator, CohereEmbeddingGenerator, get_embedding_generator

__all__ = [
    'BaseEmbeddingGenerator',
    'CohereEmbeddingGenerator',
    'get_embedding_generator'
]