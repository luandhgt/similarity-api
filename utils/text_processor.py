#!/usr/bin/env python3
"""
Text processing utilities for text embedding

Supports multiple embedding providers:
- Voyage AI (voyage-3-large, voyage-3, etc.)
- OpenAI (text-embedding-3-large, text-embedding-3-small, etc.)
- Cohere (embed-english-v3.0, etc.)

Provider selection via EMBEDDING_PROVIDER environment variable.
"""

import os
import requests
import numpy as np
from typing import List, Union, Optional
import re
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# LEGACY VOYAGE CLIENT (kept for backward compatibility)
# =============================================================================

class VoyageClient:
    """
    Legacy client for Voyage-3-Large API

    DEPRECATED: Use get_embedding_provider() instead for multi-provider support.
    Kept for backward compatibility with existing code.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('VOYAGE_API_KEY')
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY not found in environment variables")

        self.base_url = "https://api.voyageai.com/v1/embeddings"
        self.model = "voyage-3-large"

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Extract embedding vector(s) from text using Voyage-3-Large

        Args:
            text: Single text string or list of texts

        Returns:
            numpy.ndarray: Embedding vector(s)
                - Single text: shape (1024,)
                - Multiple texts: shape (n_texts, 1024)
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
            "Content-Type": "application/json"
        }

        payload = {
            "input": texts,
            "model": self.model
        }

        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()

            result = response.json()
            embeddings = [item['embedding'] for item in result['data']]

            # Convert to numpy array
            embeddings_array = np.array(embeddings)

            # Return single vector for single input
            if single_input:
                return embeddings_array.squeeze()

            return embeddings_array

        except requests.exceptions.RequestException as e:
            raise Exception(f"Voyage API request failed: {e}")
        except KeyError as e:
            raise Exception(f"Unexpected API response format: {e}")


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

def preprocess_text(text: str, lowercase: bool = False) -> str:
    """
    Preprocess text before embedding

    Args:
        text: Raw text input
        lowercase: Whether to convert to lowercase (default False - keeps original case for better semantic similarity)

    Returns:
        str: Preprocessed text
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    # Basic preprocessing
    text = text.strip()

    # Remove excessive whitespace
    text = re.sub(r'#+\s*', '', text)  # Remove headers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove bold
    text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

    # Lowercase (optional, some embeddings work better with original case)
    if lowercase:
        text = text.lower()

    return text


def validate_text_input(text: str) -> bool:
    """
    Validate text input

    Args:
        text: Input text

    Returns:
        bool: True if valid

    Raises:
        Exception: If validation fails
    """
    if not text:
        raise Exception("Text cannot be None")

    if not text.strip():
        raise Exception("Text cannot be empty or only whitespace")

    # Check reasonable length (adjust as needed)
    if len(text.strip()) < 3:
        raise Exception("Text too short (minimum 3 characters)")

    if len(text) > 100000:  # 100k chars limit
        raise Exception("Text too long (maximum 100,000 characters)")

    return True


# =============================================================================
# FEATURE EXTRACTION (supports both legacy and new provider system)
# =============================================================================

def extract_text_features(
    text: str,
    voyage_client: VoyageClient = None,
    embedding_provider=None,
    preprocess: bool = True,
    lowercase: bool = False
) -> np.ndarray:
    """
    Extract feature vector from text using configured embedding provider

    Args:
        text: Input text
        voyage_client: Optional legacy VoyageClient instance (backward compatibility)
        embedding_provider: Optional BaseEmbeddingProvider instance (new system)
        preprocess: Whether to preprocess text (default True)
        lowercase: Whether to lowercase text during preprocessing (default False)

    Returns:
        numpy.ndarray: Feature vector (dimensions depend on provider/model)
    """

    # Preprocess text if requested
    if preprocess:
        processed_text = preprocess_text(text, lowercase=lowercase)
    else:
        processed_text = text

    # Use new provider system if available
    if embedding_provider is not None:
        return embedding_provider.embed_text(processed_text)

    # Use legacy VoyageClient if provided
    if voyage_client is not None:
        return voyage_client.embed_text(processed_text)

    # Fallback: create provider from environment
    provider = get_embedding_provider()
    return provider.embed_text(processed_text)


def extract_text_features_batch(
    texts: List[str],
    embedding_provider=None,
    preprocess: bool = True,
    lowercase: bool = False,
    batch_size: int = 100
) -> np.ndarray:
    """
    Extract feature vectors for multiple texts

    Args:
        texts: List of input texts
        embedding_provider: Optional BaseEmbeddingProvider instance
        preprocess: Whether to preprocess texts (default True)
        lowercase: Whether to lowercase texts during preprocessing (default False)
        batch_size: Number of texts per API call

    Returns:
        numpy.ndarray: Feature vectors, shape (n_texts, embedding_dim)
    """
    # Preprocess texts if requested
    if preprocess:
        processed_texts = [preprocess_text(t, lowercase=lowercase) for t in texts]
    else:
        processed_texts = texts

    # Get provider
    if embedding_provider is None:
        embedding_provider = get_embedding_provider()

    return embedding_provider.embed_batch(processed_texts, batch_size=batch_size)


# =============================================================================
# PROVIDER MANAGEMENT
# =============================================================================

# Global provider instance (lazy loading)
_embedding_provider = None
_voyage_client = None  # Legacy


def get_embedding_provider():
    """
    Get singleton embedding provider instance based on EMBEDDING_PROVIDER env var

    Returns:
        BaseEmbeddingProvider instance
    """
    global _embedding_provider

    if _embedding_provider is None:
        from services.embedding_provider_factory import get_cached_embedding_provider
        _embedding_provider = get_cached_embedding_provider()
        logger.info(
            f"✅ Initialized embedding provider: "
            f"{_embedding_provider.get_provider_info()}"
        )

    return _embedding_provider


def get_voyage_client() -> VoyageClient:
    """
    Get singleton VoyageClient instance (legacy, backward compatibility)

    DEPRECATED: Use get_embedding_provider() instead.
    """
    global _voyage_client
    if _voyage_client is None:
        _voyage_client = VoyageClient()
    return _voyage_client


def reset_embedding_provider():
    """Reset the global embedding provider (useful for testing or switching providers)"""
    global _embedding_provider
    _embedding_provider = None
    logger.info("🔄 Embedding provider reset")
