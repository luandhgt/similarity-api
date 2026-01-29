"""
OpenAI Embedding Provider

Implementation of the BaseEmbeddingProvider for OpenAI's embedding API.
"""

import logging
import requests
from typing import List, Union, Dict, Any, Optional

import numpy as np

from services.embedding_provider_base import (
    BaseEmbeddingProvider,
    EmbeddingProviderType,
    EMBEDDING_PROVIDER_CONFIGS,
)

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI embedding provider implementation.

    Supports models:
        - text-embedding-3-large (3072 dimensions, best quality)
        - text-embedding-3-small (1536 dimensions, good balance)
        - text-embedding-ada-002 (1536 dimensions, legacy)

    Features:
        - Supports dimension reduction via 'dimensions' parameter
        - Native normalization
    """

    def __init__(
        self,
        api_key: str,
        model: str = None,
        dimensions: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize OpenAI embedding provider

        Args:
            api_key: OpenAI API key
            model: Model name (default: text-embedding-3-large)
            dimensions: Optional reduced dimensions (only for text-embedding-3-* models)
            **kwargs: Additional configuration
        """
        if not api_key:
            raise ValueError("OpenAI API key is required")

        self.api_key = api_key

        # Get provider config
        config = EMBEDDING_PROVIDER_CONFIGS[EmbeddingProviderType.OPENAI]
        self.base_url = config["base_url"]

        # Set model
        self.model = model or config["default_model"]

        # Validate model
        if self.model not in config["models"]:
            available = ", ".join(config["models"].keys())
            raise ValueError(
                f"Unknown OpenAI model: {self.model}. Available: {available}"
            )

        # Get model config
        model_config = config["models"][self.model]
        self._native_dimension = model_config["dimensions"]
        self._max_tokens = model_config["max_tokens"]

        # Handle dimension reduction (only for text-embedding-3-* models)
        self._requested_dimensions = dimensions
        if dimensions and self.model.startswith("text-embedding-3"):
            if dimensions > self._native_dimension:
                raise ValueError(
                    f"Requested dimensions ({dimensions}) cannot exceed "
                    f"native dimensions ({self._native_dimension})"
                )
            self._embedding_dimension = dimensions
            logger.info(
                f"📐 Using reduced dimensions: {dimensions} "
                f"(native: {self._native_dimension})"
            )
        else:
            self._embedding_dimension = self._native_dimension
            if dimensions and not self.model.startswith("text-embedding-3"):
                logger.warning(
                    f"⚠️ Dimension reduction not supported for {self.model}, "
                    f"using native {self._native_dimension}"
                )

        logger.info(
            f"✅ OpenAIEmbeddingProvider initialized: model={self.model}, "
            f"dimensions={self._embedding_dimension}"
        )

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Extract embedding vector(s) from text using OpenAI API

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

        # Add dimensions parameter if requested (for text-embedding-3-* models)
        if (
            self._requested_dimensions
            and self.model.startswith("text-embedding-3")
        ):
            payload["dimensions"] = self._requested_dimensions

        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()

            result = response.json()

            # Sort by index to maintain order
            data = sorted(result["data"], key=lambda x: x["index"])
            embeddings = [item["embedding"] for item in data]

            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)

            # Return single vector for single input
            if single_input:
                return embeddings_array.squeeze()

            return embeddings_array

        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI API request failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise Exception(f"OpenAI API request failed: {e}")
        except KeyError as e:
            logger.error(f"Unexpected OpenAI API response format: {e}")
            raise Exception(f"Unexpected API response format: {e}")

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Extract embeddings for a batch of texts with automatic batching

        OpenAI supports up to 2048 texts per request, but we use smaller
        batches for reliability.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call

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
            "provider": "openai",
            "model": self.model,
            "dimensions": self._embedding_dimension,
            "native_dimensions": self._native_dimension,
            "max_tokens": self._max_tokens,
            "status": "ready",
        }

    @property
    def provider_type(self) -> EmbeddingProviderType:
        """Return the provider type enum"""
        return EmbeddingProviderType.OPENAI

    @property
    def embedding_dimension(self) -> int:
        """Return the embedding dimension for this model"""
        return self._embedding_dimension

    @property
    def max_tokens(self) -> int:
        """Return the maximum tokens this model can handle"""
        return self._max_tokens
