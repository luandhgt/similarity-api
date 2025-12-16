"""
LLM Provider Factory

Factory class for creating and managing LLM providers (Claude, ChatGPT, Gemini, etc.)
Provides easy switching between different LLM providers based on configuration.
"""
import logging
from typing import Dict, Any, Optional

from services.llm_provider_base import BaseLLMProvider, LLMProviderType
from services.claude_provider import ClaudeProvider
from services.chatgpt_provider import ChatGPTProvider

logger = logging.getLogger(__name__)


class LLMProviderFactory:
    """
    Factory for creating LLM provider instances

    Usage:
        # Create provider from config
        provider = LLMProviderFactory.create_provider(
            provider_type="claude",
            api_key="sk-...",
            model="claude-3-5-sonnet-20241022"
        )

        # Use provider
        response = await provider.generate_text("Hello, world!")
    """

    # Registry of available providers
    _PROVIDER_CLASSES = {
        LLMProviderType.CLAUDE: ClaudeProvider,
        LLMProviderType.CHATGPT: ChatGPTProvider,
        # Add more providers here as they are implemented
        # LLMProviderType.GEMINI: GeminiProvider,
    }

    # Default models for each provider
    _DEFAULT_MODELS = {
        LLMProviderType.CLAUDE: "claude-sonnet-4-5-20250929",
        LLMProviderType.CHATGPT: "gpt-4o",
        LLMProviderType.GEMINI: "gemini-pro",
    }

    @classmethod
    def create_provider(
        cls,
        provider_type: str,
        api_key: str,
        model: Optional[str] = None,
        **kwargs
    ) -> BaseLLMProvider:
        """
        Create an LLM provider instance

        Args:
            provider_type: Provider type ("claude", "chatgpt", "gemini")
            api_key: API key for the provider
            model: Model identifier (optional, uses default if not provided)
            **kwargs: Additional provider-specific configuration

        Returns:
            Initialized LLM provider instance

        Raises:
            ValueError: If provider_type is not supported
        """
        # Convert string to enum
        try:
            provider_enum = LLMProviderType(provider_type.lower())
        except ValueError:
            supported = ", ".join([p.value for p in LLMProviderType])
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

        # Use default model if not provided
        if not model:
            model = cls._DEFAULT_MODELS.get(provider_enum)
            logger.info(f"📝 Using default model for {provider_type}: {model}")

        # Create and return provider instance
        logger.info(f"🚀 Creating {provider_type} provider with model: {model}")
        return provider_class(api_key=api_key, model=model, **kwargs)

    @classmethod
    def create_provider_from_config(cls, config: Dict[str, Any]) -> BaseLLMProvider:
        """
        Create provider from configuration dictionary

        Args:
            config: Configuration dictionary with keys:
                - provider_type: "claude", "chatgpt", etc.
                - api_key: API key
                - model: Model name (optional)
                - max_tokens: Max tokens (optional)
                - temperature: Temperature (optional)
                - timeout: Timeout in seconds (optional)

        Returns:
            Initialized LLM provider instance
        """
        provider_type = config.get("provider_type")
        if not provider_type:
            raise ValueError("provider_type is required in config")

        api_key = config.get("api_key")
        if not api_key:
            raise ValueError("api_key is required in config")

        # Extract provider-specific kwargs
        kwargs = {
            k: v for k, v in config.items()
            if k not in ["provider_type", "api_key"]
        }

        return cls.create_provider(
            provider_type=provider_type,
            api_key=api_key,
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
            provider_enum = LLMProviderType(provider_type.lower())
            return cls._DEFAULT_MODELS.get(provider_enum)
        except ValueError:
            return None

    @classmethod
    def validate_provider_config(cls, config: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate provider configuration

        Args:
            config: Configuration dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required fields
        if "provider_type" not in config:
            return False, "provider_type is required"

        if "api_key" not in config:
            return False, "api_key is required"

        # Check provider type is supported
        provider_type = config["provider_type"]
        if provider_type not in cls.get_supported_providers():
            supported = ", ".join(cls.get_supported_providers())
            return False, f"Unsupported provider: {provider_type}. Supported: {supported}"

        return True, ""


# Singleton instance cache
_provider_cache: Dict[str, BaseLLMProvider] = {}


def get_cached_provider(
    provider_type: str,
    api_key: str,
    model: Optional[str] = None,
    **kwargs
) -> BaseLLMProvider:
    """
    Get or create a cached provider instance

    This function maintains a singleton instance of each provider type
    to avoid recreating providers unnecessarily.

    Args:
        provider_type: Provider type ("claude", "chatgpt", "gemini")
        api_key: API key for the provider
        model: Model identifier (optional)
        **kwargs: Additional provider-specific configuration

    Returns:
        Cached or newly created provider instance
    """
    cache_key = f"{provider_type}:{model or 'default'}"

    if cache_key not in _provider_cache:
        logger.info(f"📦 Creating new provider instance: {cache_key}")
        _provider_cache[cache_key] = LLMProviderFactory.create_provider(
            provider_type=provider_type,
            api_key=api_key,
            model=model,
            **kwargs
        )
    else:
        logger.debug(f"♻️  Reusing cached provider: {cache_key}")

    return _provider_cache[cache_key]


def clear_provider_cache():
    """Clear the provider cache (useful for testing)"""
    global _provider_cache
    _provider_cache.clear()
    logger.info("🧹 Provider cache cleared")
