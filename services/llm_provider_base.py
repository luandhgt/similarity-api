"""
Base LLM Provider Interface

Abstract base class defining the interface that all LLM providers must implement.
This allows easy switching between different LLM providers (Claude, ChatGPT, Gemini, etc.)
"""
import base64
import io
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


class LLMProviderType(Enum):
    """Supported LLM provider types"""
    CLAUDE = "claude"
    CHATGPT = "chatgpt"
    GEMINI = "gemini"


# Image size limits for different providers
IMAGE_LIMITS = {
    LLMProviderType.CLAUDE: {
        "max_pixels": 1568,      # Max dimension (width or height)
        "max_bytes": 5 * 1024 * 1024,  # 5MB
    },
    LLMProviderType.CHATGPT: {
        "max_pixels": 2048,      # Max dimension
        "max_bytes": 20 * 1024 * 1024,  # 20MB
    },
    LLMProviderType.GEMINI: {
        "max_pixels": 3072,      # Max dimension
        "max_bytes": 20 * 1024 * 1024,  # 20MB
    },
}


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.

    All LLM providers must implement these methods to ensure consistent interface
    across different AI services (Claude, ChatGPT, Gemini, etc.)
    """

    def _get_image_limits(self) -> dict:
        """Get image size limits for this provider"""
        return IMAGE_LIMITS.get(self.provider_type, IMAGE_LIMITS[LLMProviderType.CLAUDE])

    def _resize_image_if_needed(self, image_path: str) -> Tuple[bytes, str, bool]:
        """
        Resize image if it exceeds provider limits.

        Args:
            image_path: Path to original image

        Returns:
            Tuple of (image_bytes, media_type, was_resized)
        """
        limits = self._get_image_limits()
        max_pixels = limits["max_pixels"]
        max_bytes = limits["max_bytes"]

        # Get file extension for media type
        ext = Path(image_path).suffix.lower()
        media_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        media_type = media_types.get(ext, 'image/jpeg')

        # Read original file
        with open(image_path, 'rb') as f:
            original_bytes = f.read()

        original_size = len(original_bytes)

        # Check if resize is needed
        img = Image.open(io.BytesIO(original_bytes))
        width, height = img.size

        needs_resize = (
            width > max_pixels or
            height > max_pixels or
            original_size > max_bytes
        )

        if not needs_resize:
            logger.debug(f"✅ Image {Path(image_path).name} within limits ({width}x{height}, {original_size/1024:.1f}KB)")
            return original_bytes, media_type, False

        logger.info(f"🔄 Resizing {Path(image_path).name}: {width}x{height} ({original_size/1024/1024:.2f}MB) -> max {max_pixels}px, <{max_bytes/1024/1024}MB")

        # Convert to RGB if necessary (for PNG with transparency)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
            media_type = 'image/jpeg'  # Convert to JPEG for smaller size

        # Resize if dimensions exceed limit
        if width > max_pixels or height > max_pixels:
            ratio = min(max_pixels / width, max_pixels / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"   📐 Resized to {new_width}x{new_height}")

        # Compress with decreasing quality until under max_bytes
        quality = 95
        while quality >= 20:
            buffer = io.BytesIO()
            if media_type == 'image/png':
                img.save(buffer, format='PNG', optimize=True)
            else:
                img.save(buffer, format='JPEG', quality=quality, optimize=True)

            result_bytes = buffer.getvalue()
            if len(result_bytes) <= max_bytes:
                logger.info(f"   ✅ Compressed to {len(result_bytes)/1024:.1f}KB (quality={quality})")
                return result_bytes, media_type, True

            quality -= 10

        # If still too large, return best effort
        logger.warning(f"   ⚠️ Could not compress below {max_bytes/1024/1024}MB, using {len(result_bytes)/1024/1024:.2f}MB")
        return result_bytes, media_type, True

    def _encode_image_with_resize(self, image_path: str) -> Tuple[str, str]:
        """
        Encode image to base64, resizing if needed.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (base64_string, media_type)
        """
        image_bytes, media_type, was_resized = self._resize_image_if_needed(image_path)
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        return base64_string, media_type

    @abstractmethod
    def __init__(self, api_key: str, model: str, **kwargs):
        """
        Initialize the LLM provider

        Args:
            api_key: API key for authentication
            model: Model identifier (e.g., 'claude-3-5-sonnet', 'gpt-4', 'gemini-pro')
            **kwargs: Additional provider-specific configuration
        """
        pass

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate text from a simple text prompt

        Args:
            prompt: User prompt
            system_prompt: System/context prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    async def analyze_image(
        self,
        image_path: str,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Analyze a single image with text prompt

        Args:
            image_path: Path to image file
            prompt: User prompt for analysis
            system_prompt: System/context prompt

        Returns:
            Analysis result text
        """
        pass

    @abstractmethod
    async def analyze_multiple_images(
        self,
        image_paths: List[str],
        prompts: List[str],
        system_prompt: Optional[str] = None,
        parallel: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple images (OCR/extraction workflow)

        Args:
            image_paths: List of image file paths
            prompts: List of prompts (one per image)
            system_prompt: System/context prompt
            parallel: Whether to process in parallel

        Returns:
            List of analysis results with success status
        """
        pass

    @abstractmethod
    async def process_multiple_images(
        self,
        image_paths: List[str],
        system_prompt: str = "",
        user_prompt: str = "",
        parallel: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images for OCR (Claude-compatible method)

        Args:
            image_paths: List of image file paths
            system_prompt: System prompt for OCR
            user_prompt: User prompt for OCR
            parallel: Whether to process images in parallel

        Returns:
            List of results with image path, extracted text, and status
        """
        pass

    @abstractmethod
    async def synthesize_content(
        self,
        texts: List[str],
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None
    ) -> str:
        """
        Synthesize multiple texts into coherent content

        Args:
            texts: List of texts to synthesize
            system_prompt: System/context prompt
            user_prompt: User instructions for synthesis

        Returns:
            Synthesized content
        """
        pass

    @abstractmethod
    async def synthesize_with_images_and_texts(
        self,
        image_paths: List[str],
        texts: List[str],
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None
    ) -> str:
        """
        Synthesize content from both images and texts (multimodal analysis)

        Args:
            image_paths: List of image file paths
            texts: List of texts (e.g., OCR results)
            system_prompt: System/context prompt
            user_prompt: User instructions for synthesis

        Returns:
            Synthesized content with visual and textual analysis
        """
        pass

    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get provider information and statistics

        Returns:
            Dictionary with provider name, model, status, etc.
        """
        pass

    @property
    @abstractmethod
    def provider_type(self) -> LLMProviderType:
        """Return the provider type enum"""
        pass

    @property
    @abstractmethod
    def supports_vision(self) -> bool:
        """Return whether this provider supports vision/image analysis"""
        pass
