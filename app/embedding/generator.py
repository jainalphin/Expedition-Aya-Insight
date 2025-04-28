# embedding/generator.py

"""
Embedding generation module for creating vector embeddings from text.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Setup module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class BaseEmbeddingGenerator:
    """Base class for text embedding generators"""

    def __init__(self):
        """Initialize the base embedding generator"""
        pass

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of embeddings
        """
        raise NotImplementedError("Subclasses must implement this method")

    def validate_texts(self, texts: List[str]) -> None:
        """
        Validate that texts aren't empty.

        Args:
            texts: List of text strings to validate

        Raises:
            ValueError: If any text is empty
        """
        for i, text in enumerate(texts):
            if not text or text.strip() == "":
                error_msg = f"Cannot create embedding for empty text at index {i}"
                logger.error(error_msg)
                raise ValueError(error_msg)


class CohereEmbeddingGenerator(BaseEmbeddingGenerator):
    """Text embedding generator using Cohere API"""

    def __init__(
            self,
            model: str = "embed-v4.0",
            embedding_dimension: int = 1024,
            api_key_env_var: str = "COHERE_API_KEY",
            input_type: str = "search_document"
    ):
        """
        Initialize the Cohere embedding generator.

        Args:
            model: Cohere embedding model to use
            embedding_dimension: Output dimension for embeddings
            api_key_env_var: Environment variable name for Cohere API key
            input_type: Type of input for Cohere embeddings
        """
        super().__init__()

        # Load environment variables
        load_dotenv()

        # Check for API key
        if not os.getenv(api_key_env_var):
            logger.error(f"Missing Cohere API key in environment variable: {api_key_env_var}")
            raise ValueError(f"Missing Cohere API key. Set the {api_key_env_var} environment variable.")

        self.model = model
        self.embedding_dimension = embedding_dimension
        self.api_key = os.getenv(api_key_env_var)
        self.input_type = input_type

        # Initialize Cohere client
        try:
            import cohere
            self.client = cohere.ClientV2(self.api_key)
            logger.info("Cohere client initialized successfully")
        except ImportError:
            logger.error("Failed to import Cohere. Make sure it's installed using 'pip install cohere'")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Cohere client: {str(e)}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts using Cohere.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of embeddings

        Raises:
            ValueError: If any text is empty
            Exception: If Cohere API call fails
        """
        # Validate texts
        self.validate_texts(texts)

        logger.info(f"Generating embeddings for {len(texts)} texts using Cohere")

        try:
            response = self.client.embed(
                texts=texts,
                model=self.model,
                input_type=self.input_type,
                output_dimension=self.embedding_dimension,
                embedding_types=["float"],
            )

            embeddings = np.array(response.embeddings.float_)
            logger.info(f"Successfully generated embeddings of shape {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise


# Factory function to get appropriate embedding generator
def get_embedding_generator(provider: str = "cohere", **kwargs) -> BaseEmbeddingGenerator:
    """
    Factory function to get an embedding generator.

    Args:
        provider: The embedding provider to use ('cohere' supported)
        **kwargs: Additional arguments to pass to the generator

    Returns:
        An embedding generator instance

    Raises:
        ValueError: If provider is not supported
    """
    if provider.lower() == "cohere":
        return CohereEmbeddingGenerator(**kwargs)
    else:
        error_msg = f"Unsupported embedding provider: {provider}"
        logger.error(error_msg)
        raise ValueError(error_msg)