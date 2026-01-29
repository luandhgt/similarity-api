"""
Base Embedding Provider Interface

Abstract base class defining the interface that all embedding providers must implement.
This allows easy switching between different embedding providers (Voyage, OpenAI, Cohere, etc.)
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Any
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProviderType(Enum):
    """Supported embedding provider types"""
    VOYAGE = "voyage"
    OPENAI = "openai"
    COHERE = "cohere"


class BaseEmbeddingProvider(ABC):
    """
    Abstract base class for all embedding providers.

    All embedding providers must implement these methods to ensure consistent interface
    across different embedding services (Voyage, OpenAI, Cohere, etc.)
    """

    @abstractmethod
    def __init__(self, api_key: str, model: str, **kwargs):
        """
        Initialize the embedding provider

        Args:
            api_key: API key for authentication
            model: Model identifier (e.g., 'voyage-3-large', 'text-embedding-3-large')
            **kwargs: Additional provider-specific configuration
        """
        pass

    @abstractmethod
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Extract embedding vector(s) from text

        Args:
            text: Single text string or list of texts

        Returns:
            numpy.ndarray: Embedding vector(s)
                - Single text: shape (embedding_dim,)
                - Multiple texts: shape (n_texts, embedding_dim)
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Extract embeddings for a batch of texts with automatic batching

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call

        Returns:
            numpy.ndarray: Embedding vectors, shape (n_texts, embedding_dim)
        """
        pass

    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get provider information and configuration

        Returns:
            Dictionary with provider name, model, dimensions, etc.
        """
        pass

    @property
    @abstractmethod
    def provider_type(self) -> EmbeddingProviderType:
        """Return the provider type enum"""
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the embedding dimension for this model"""
        pass

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """Return the maximum tokens this model can handle per text"""
        pass


# Provider capabilities and default configurations
EMBEDDING_PROVIDER_CONFIGS = {
    EmbeddingProviderType.VOYAGE: {
        "default_model": "voyage-3-large",
        "models": {
            "voyage-3-large": {"dimensions": 1024, "max_tokens": 32000},
            "voyage-3": {"dimensions": 1024, "max_tokens": 32000},
            "voyage-3-lite": {"dimensions": 512, "max_tokens": 32000},
            "voyage-3.5-lite": {"dimensions": 1024, "max_tokens": 32000},
        },
        "base_url": "https://api.voyageai.com/v1/embeddings",
    },
    EmbeddingProviderType.OPENAI: {
        "default_model": "text-embedding-3-large",
        "models": {
            "text-embedding-3-large": {"dimensions": 3072, "max_tokens": 8191},
            "text-embedding-3-small": {"dimensions": 1536, "max_tokens": 8191},
            "text-embedding-ada-002": {"dimensions": 1536, "max_tokens": 8191},
        },
        "base_url": "https://api.openai.com/v1/embeddings",
    },
    EmbeddingProviderType.COHERE: {
        "default_model": "embed-english-v3.0",
        "models": {
            "embed-english-v3.0": {"dimensions": 1024, "max_tokens": 512},
            "embed-multilingual-v3.0": {"dimensions": 1024, "max_tokens": 512},
            "embed-english-light-v3.0": {"dimensions": 384, "max_tokens": 512},
        },
        "base_url": "https://api.cohere.ai/v1/embed",
    },
}
