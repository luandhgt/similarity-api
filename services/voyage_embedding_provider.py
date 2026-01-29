"""
Voyage AI Embedding Provider

Implementation of the BaseEmbeddingProvider for Voyage AI's embedding API.
"""

import logging
import requests
from typing import List, Union, Dict, Any

import numpy as np

from services.embedding_provider_base import (
    BaseEmbeddingProvider,
    EmbeddingProviderType,
    EMBEDDING_PROVIDER_CONFIGS,
)

logger = logging.getLogger(__name__)


class VoyageEmbeddingProvider(BaseEmbeddingProvider):
    """
    Voyage AI embedding provider implementation.

    Supports models:
        - voyage-3-large (1024 dimensions)
        - voyage-3 (1024 dimensions)
        - voyage-3-lite (512 dimensions)
        - voyage-3.5-lite (1024 dimensions)
    """

    def __init__(self, api_key: str, model: str = None, **kwargs):
        """
        Initialize Voyage embedding provider

        Args:
            api_key: Voyage AI API key
            model: Model name (default: voyage-3-large)
            **kwargs: Additional configuration
        """
        if not api_key:
            raise ValueError("Voyage API key is required")

        self.api_key = api_key

        # Get provider config
        config = EMBEDDING_PROVIDER_CONFIGS[EmbeddingProviderType.VOYAGE]
        self.base_url = config["base_url"]

        # Set model
        self.model = model or config["default_model"]

        # Validate model
        if self.model not in config["models"]:
            available = ", ".join(config["models"].keys())
            raise ValueError(
                f"Unknown Voyage model: {self.model}. Available: {available}"
            )

        # Get model config
        model_config = config["models"][self.model]
        self._embedding_dimension = model_config["dimensions"]
        self._max_tokens = model_config["max_tokens"]

        logger.info(
            f"✅ VoyageEmbeddingProvider initialized: model={self.model}, "
            f"dimensions={self._embedding_dimension}"
        )

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Extract embedding vector(s) from text using Voyage API

        Args:
            text: Single text string or list of texts

        Returns:
            numpy.ndarray: Embedding vector(s)
        """
        # Ensure text is a list
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False

        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "input": texts,
            "model": self.model,
        }

        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()

            result = response.json()
            embeddings = [item["embedding"] for item in result["data"]]

            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)

            # Return single vector for single input
            if single_input:
                return embeddings_array.squeeze()

            return embeddings_array

        except requests.exceptions.RequestException as e:
            logger.error(f"Voyage API request failed: {e}")
            raise Exception(f"Voyage API request failed: {e}")
        except KeyError as e:
            logger.error(f"Unexpected Voyage API response format: {e}")
            raise Exception(f"Unexpected API response format: {e}")

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Extract embeddings for a batch of texts with automatic batching

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call (Voyage supports up to 128)

        Returns:
            numpy.ndarray: Embedding vectors, shape (n_texts, embedding_dim)
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.debug(f"Processing batch {i // batch_size + 1}: {len(batch)} texts")

            embeddings = self.embed_text(batch)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information"""
        return {
            "provider": "voyage",
            "model": self.model,
            "dimensions": self._embedding_dimension,
            "max_tokens": self._max_tokens,
            "status": "ready",
        }

    @property
    def provider_type(self) -> EmbeddingProviderType:
        """Return the provider type enum"""
        return EmbeddingProviderType.VOYAGE

    @property
    def embedding_dimension(self) -> int:
        """Return the embedding dimension for this model"""
        return self._embedding_dimension

    @property
    def max_tokens(self) -> int:
        """Return the maximum tokens this model can handle"""
        return self._max_tokens
