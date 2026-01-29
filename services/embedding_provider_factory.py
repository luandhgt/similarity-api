"""
Embedding Provider Factory

Factory class for creating and managing embedding providers (Voyage, OpenAI, Cohere, etc.)
Provides easy switching between different embedding providers based on configuration.
"""

import os
import logging
from typing import Dict, Any, Optional

from services.embedding_provider_base import (
    BaseEmbeddingProvider,
    EmbeddingProviderType,
    EMBEDDING_PROVIDER_CONFIGS,
)
from services.voyage_embedding_provider import VoyageEmbeddingProvider
from services.openai_embedding_provider import OpenAIEmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingProviderFactory:
    """
    Factory for creating embedding provider instances

    Usage:
        # Create provider from config
        provider = EmbeddingProviderFactory.create_provider(
            provider_type="openai",
            api_key="sk-...",
            model="text-embedding-3-large"
        )

        # Use provider
        vector = provider.embed_text("Hello, world!")
    """

    # Registry of available providers
    _PROVIDER_CLASSES = {
        EmbeddingProviderType.VOYAGE: VoyageEmbeddingProvider,
        EmbeddingProviderType.OPENAI: OpenAIEmbeddingProvider,
        # Add more providers here as they are implemented
        # EmbeddingProviderType.COHERE: CohereEmbeddingProvider,
    }

    # API key environment variable names for each provider
    _API_KEY_ENV_VARS = {
        EmbeddingProviderType.VOYAGE: "VOYAGE_API_KEY",
        EmbeddingProviderType.OPENAI: "OPENAI_API_KEY",
        EmbeddingProviderType.COHERE: "COHERE_API_KEY",
    }

    @classmethod
    def create_provider(
        cls,
        provider_type: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> BaseEmbeddingProvider:
        """
        Create an embedding provider instance

        Args:
            provider_type: Provider type ("voyage", "openai", "cohere")
            api_key: API key for the provider (optional, will use env var if not provided)
            model: Model identifier (optional, uses default if not provided)
            **kwargs: Additional provider-specific configuration

        Returns:
            Initialized embedding provider instance

        Raises:
            ValueError: If provider_type is not supported or API key is missing
        """
        # Convert string to enum
        try:
            provider_enum = EmbeddingProviderType(provider_type.lower())
        except ValueError:
            supported = ", ".join([p.value for p in EmbeddingProviderType])
            raise ValueError(
                f"Unsupported provider type: {provider_type}. "
                f"Supported providers: {supported}"
            )

        # Get provider class
        provider_class = cls._PROVIDER_CLASSES.get(provider_enum)
        if not provider_class:
            raise ValueError(
                f"Provider {provider_type} is defined but not yet implemented. "
                f"Available providers: {', '.join([p.value for p in cls._PROVIDER_CLASSES.keys()])}"
            )

        # Get API key from env if not provided
        if not api_key:
            env_var = cls._API_KEY_ENV_VARS.get(provider_enum)
            api_key = os.getenv(env_var) if env_var else None

            if not api_key:
                raise ValueError(
                    f"API key required for {provider_type}. "
                    f"Set {env_var} environment variable or provide api_key parameter."
                )

        # Use default model if not provided
        if not model:
            config = EMBEDDING_PROVIDER_CONFIGS.get(provider_enum, {})
            model = config.get("default_model")
            logger.info(f"📝 Using default model for {provider_type}: {model}")

        # Create and return provider instance
        logger.info(f"🚀 Creating {provider_type} embedding provider with model: {model}")
        return provider_class(api_key=api_key, model=model, **kwargs)

    @classmethod
    def create_provider_from_env(cls, **kwargs) -> BaseEmbeddingProvider:
        """
        Create embedding provider from environment variables

        Environment variables:
            - EMBEDDING_PROVIDER: Provider type ("voyage", "openai", "cohere")
            - EMBEDDING_MODEL: Model name (optional)
            - VOYAGE_API_KEY / OPENAI_API_KEY / COHERE_API_KEY: API key

        Args:
            **kwargs: Additional provider-specific configuration

        Returns:
            Initialized embedding provider instance
        """
        provider_type = os.getenv("EMBEDDING_PROVIDER", "voyage")
        model = os.getenv("EMBEDDING_MODEL")

        logger.info(f"📦 Creating embedding provider from env: {provider_type}")

        return cls.create_provider(
            provider_type=provider_type,
            model=model,
            **kwargs
        )

    @classmethod
    def get_supported_providers(cls) -> list:
        """
        Get list of supported provider types

        Returns:
            List of provider type strings
        """
        return [provider.value for provider in cls._PROVIDER_CLASSES.keys()]

    @classmethod
    def get_default_model(cls, provider_type: str) -> Optional[str]:
        """
        Get default model for a provider type

        Args:
            provider_type: Provider type string

        Returns:
            Default model name or None if provider not found
        """
        try:
            provider_enum = EmbeddingProviderType(provider_type.lower())
            config = EMBEDDING_PROVIDER_CONFIGS.get(provider_enum, {})
            return config.get("default_model")
        except ValueError:
            return None

    @classmethod
    def get_model_info(cls, provider_type: str, model: str) -> Optional[Dict[str, Any]]:
        """
        Get model information (dimensions, max_tokens, etc.)

        Args:
            provider_type: Provider type string
            model: Model name

        Returns:
            Model info dict or None if not found
        """
        try:
            provider_enum = EmbeddingProviderType(provider_type.lower())
            config = EMBEDDING_PROVIDER_CONFIGS.get(provider_enum, {})
            models = config.get("models", {})
            return models.get(model)
        except ValueError:
            return None


# Singleton instance cache
_provider_cache: Dict[str, BaseEmbeddingProvider] = {}


def get_cached_embedding_provider(
    provider_type: str = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> BaseEmbeddingProvider:
    """
    Get or create a cached embedding provider instance

    This function maintains a singleton instance of each provider type
    to avoid recreating providers unnecessarily.

    Args:
        provider_type: Provider type ("voyage", "openai", "cohere") - default from env
        api_key: API key for the provider (optional)
        model: Model identifier (optional)
        **kwargs: Additional provider-specific configuration

    Returns:
        Cached or newly created provider instance
    """
    # Get provider type from env if not specified
    if not provider_type:
        provider_type = os.getenv("EMBEDDING_PROVIDER", "voyage")

    cache_key = f"{provider_type}:{model or 'default'}"

    if cache_key not in _provider_cache:
        logger.info(f"📦 Creating new embedding provider instance: {cache_key}")
        _provider_cache[cache_key] = EmbeddingProviderFactory.create_provider(
            provider_type=provider_type,
            api_key=api_key,
            model=model,
            **kwargs
        )
    else:
        logger.debug(f"♻️  Reusing cached embedding provider: {cache_key}")

    return _provider_cache[cache_key]


def clear_embedding_provider_cache():
    """Clear the provider cache (useful for testing)"""
    global _provider_cache
    _provider_cache.clear()
    logger.info("🧹 Embedding provider cache cleared")
