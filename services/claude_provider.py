"""
Claude Provider - Anthropic API Integration

Refactored implementation of ClaudeService using the BaseLLMProvider interface.
This maintains backward compatibility while following the new provider pattern.
"""
import base64
import asyncio
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import aiohttp

from services.llm_provider_base import BaseLLMProvider, LLMProviderType

logger = logging.getLogger(__name__)


class ClaudeProvider(BaseLLMProvider):
    """
    Anthropic Claude provider implementation

    Supports:
    - Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku
    - Vision capabilities for image analysis
    - Multimodal content (text + images)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 8000,
        temperature: float = 0.7,
        timeout: int = 300,
        **kwargs
    ):
        """
        Initialize Claude provider

        Args:
            api_key: Anthropic API key
            model: Model name (e.g., 'claude-sonnet-4-5-20250929', 'claude-3-opus-20240229')
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            timeout: Request timeout in seconds
            **kwargs: Additional parameters
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.base_url = "https://api.anthropic.com/v1"

        if not self.api_key:
            logger.warning("⚠️  CLAUDE_API_KEY not found in environment variables")
            logger.warning("⚠️  Claude provider will not be functional until API key is provided")
        else:
            logger.info(f"✅ Claude provider initialized with model: {self.model}")

    @property
    def provider_type(self) -> LLMProviderType:
        """Return provider type"""
        return LLMProviderType.CLAUDE

    @property
    def supports_vision(self) -> bool:
        """Claude models support vision"""
        return True

    async def _make_request_with_messages(
        self,
        messages: List[Dict],
        system_prompt: str = "",
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Make async request to Claude API

        Args:
            messages: List of message objects
            system_prompt: System prompt for the conversation
            max_tokens: Override max tokens

        Returns:
            Claude's response text
        """
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": messages
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"❌ Claude API error {response.status}: {error_text}")
                        raise Exception(f"Claude API error {response.status}: {error_text}")

                    result = await response.json()
                    content = result["content"][0]["text"]

                    logger.debug(f"✅ Claude API response received (length: {len(content)})")
                    return content

        except asyncio.TimeoutError:
            logger.error("❌ Claude API request timeout")
            raise Exception("Claude API request timeout")
        except Exception as e:
            logger.error(f"❌ Claude API request failed: {e}")
            raise

    def _encode_image(self, image_path: str) -> tuple:
        """
        Encode image to base64 for Claude API (with auto-resize if needed)

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (base64_string, media_type)
        """
        try:
            # Use base class method with auto-resize
            return self._encode_image_with_resize(image_path)
        except Exception as e:
            logger.error(f"❌ Error encoding image {image_path}: {e}")
            raise

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
            temperature: Sampling temperature (not used by Claude currently)

        Returns:
            Generated text response
        """
        messages = [{
            "role": "user",
            "content": prompt
        }]

        return await self._make_request_with_messages(
            messages,
            system_prompt or "",
            max_tokens
        )

    async def analyze_image(
        self,
        image_path: str,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Analyze a single image with text prompt (OCR/extraction)

        Args:
            image_path: Path to image file
            prompt: User prompt for analysis
            system_prompt: System/context prompt

        Returns:
            Analysis result text
        """
        try:
            # Encode image (with auto-resize if needed)
            image_base64, media_type = self._encode_image(image_path)

            # Prepare message with image
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt or "Please extract all text from this image."
                    }
                ]
            }]

            # Make API request
            response = await self._make_request_with_messages(messages, system_prompt or "")

            logger.info(
                f"✅ Image analysis completed for {Path(image_path).name} "
                f"(extracted {len(response)} characters)"
            )
            return response

        except Exception as e:
            logger.error(f"❌ Image analysis failed for {image_path}: {e}")
            raise

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
            prompts: List of prompts (one per image or one for all)
            system_prompt: System/context prompt
            parallel: Whether to process in parallel

        Returns:
            List of analysis results with success status
        """
        results = []

        # If single prompt provided, use it for all images
        if len(prompts) == 1:
            prompts = prompts * len(image_paths)

        if parallel:
            # Process all images concurrently
            tasks = []
            for image_path, prompt in zip(image_paths, prompts):
                task = self._analyze_single_image_safe(image_path, prompt, system_prompt)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Convert exceptions to error results
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    final_results.append({
                        "image_path": image_paths[i],
                        "text": "",
                        "success": False,
                        "error": str(result)
                    })
                else:
                    final_results.append(result)

            return final_results
        else:
            # Process images sequentially
            for image_path, prompt in zip(image_paths, prompts):
                result = await self._analyze_single_image_safe(image_path, prompt, system_prompt)
                results.append(result)

            return results

    async def _analyze_single_image_safe(
        self,
        image_path: str,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Safely analyze a single image (with error handling)

        Returns:
            Result dictionary with success status
        """
        try:
            text = await self.analyze_image(image_path, prompt, system_prompt)
            return {
                "image_path": image_path,
                "text": text,
                "success": True,
                "error": None
            }
        except Exception as e:
            logger.error(f"❌ Failed to process {image_path}: {e}")
            return {
                "image_path": image_path,
                "text": "",
                "success": False,
                "error": str(e)
            }

    async def process_multiple_images(
        self,
        image_paths: List[str],
        system_prompt: str = "",
        user_prompt: str = "",
        parallel: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images for OCR (Claude-compatible method)

        This is a wrapper around analyze_multiple_images() with a specific signature
        that matches the original Claude implementation.

        Args:
            image_paths: List of image file paths
            system_prompt: System prompt for OCR
            user_prompt: User prompt for OCR
            parallel: Whether to process images in parallel

        Returns:
            List of results with image path, extracted text, and status
        """
        # Use the same prompt for all images
        prompts = [user_prompt or "Please extract all text from this image."] * len(image_paths)

        # Call analyze_multiple_images with the prompts
        return await self.analyze_multiple_images(
            image_paths=image_paths,
            prompts=prompts,
            system_prompt=system_prompt,
            parallel=parallel
        )

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
        try:
            # Prepare messages
            messages = [{
                "role": "user",
                "content": user_prompt or ""
            }]

            # Make API request
            response = await self._make_request_with_messages(messages, system_prompt or "")

            logger.info(f"✅ Content synthesis completed (output: {len(response)} characters)")
            return response

        except Exception as e:
            logger.error(f"❌ Content synthesis failed: {e}")
            raise

    async def synthesize_with_images_and_texts(
        self,
        image_paths: List[str],
        texts: List[str],
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None
    ) -> str:
        """
        Synthesize content from both images and texts (v2.0 workflow)

        This method sends BOTH the actual images and their OCR texts to Claude,
        allowing it to perform visual analysis alongside text analysis.

        Args:
            image_paths: List of image file paths to analyze
            texts: List of texts corresponding to images (e.g., OCR results)
            system_prompt: System prompt for analysis/synthesis
            user_prompt: User prompt with context and instructions

        Returns:
            Synthesized content with visual and textual analysis
        """
        try:
            logger.info(f"🔄 Synthesizing with {len(image_paths)} images + texts...")

            # Build content array with alternating images and texts
            content_blocks = []

            # Add all images with their texts
            for i, (image_path, text) in enumerate(zip(image_paths, texts), 1):
                # Add image (with auto-resize if needed)
                image_base64, media_type = self._encode_image(image_path)

                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_base64
                    }
                })

                # Add text context for this image
                if text:
                    content_blocks.append({
                        "type": "text",
                        "text": f"[Image {i} extracted text]:\n{text}\n"
                    })

            # Add the main user prompt at the end
            if user_prompt:
                content_blocks.append({
                    "type": "text",
                    "text": user_prompt
                })

            # Prepare message
            messages = [{
                "role": "user",
                "content": content_blocks
            }]

            # Make API request
            response = await self._make_request_with_messages(messages, system_prompt or "")

            logger.info(f"✅ Image+Text synthesis completed (output: {len(response)} characters)")
            return response

        except Exception as e:
            logger.error(f"❌ Image+Text synthesis failed: {e}")
            raise

    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information and statistics"""
        return {
            "provider": "Claude (Anthropic)",
            "provider_type": self.provider_type.value,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "supports_vision": self.supports_vision,
            "status": "ready" if self.api_key else "missing_api_key"
        }


# Backward compatibility: Keep old ClaudeService class name
ClaudeService = ClaudeProvider
