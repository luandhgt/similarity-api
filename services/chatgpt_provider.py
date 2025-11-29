"""
ChatGPT Provider - OpenAI API Integration

Implementation of BaseLLMProvider for OpenAI's ChatGPT models.
Supports GPT-4, GPT-4 Vision, and other OpenAI models.
"""
import base64
import asyncio
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import aiohttp

from services.llm_provider_base import BaseLLMProvider, LLMProviderType

logger = logging.getLogger(__name__)


class ChatGPTProvider(BaseLLMProvider):
    """
    OpenAI ChatGPT provider implementation

    Supports:
    - GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
    - GPT-4 Vision for image analysis
    - Multimodal content (text + images)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: int = 300,
        **kwargs
    ):
        """
        Initialize ChatGPT provider

        Args:
            api_key: OpenAI API key
            model: Model name (e.g., 'gpt-4o', 'gpt-4-vision-preview', 'gpt-3.5-turbo')
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            timeout: Request timeout in seconds
            **kwargs: Additional parameters
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.base_url = "https://api.openai.com/v1"

        # Check if model supports vision
        # GPT-5 models all support vision (gpt-5, gpt-5-chat, gpt-5-pro, gpt-5-mini, gpt-5-nano)
        self._supports_vision = (
            "vision" in model.lower() or
            model in ["gpt-4o", "gpt-4-turbo"] or
            model.startswith("gpt-5")
        )

        if not self.api_key:
            logger.warning("⚠️  OPENAI_API_KEY not found in environment variables")
            logger.warning("⚠️  ChatGPT provider will not be functional until API key is provided")
        else:
            logger.info(f"✅ ChatGPT provider initialized with model: {self.model}")

    @property
    def provider_type(self) -> LLMProviderType:
        """Return provider type"""
        return LLMProviderType.CHATGPT

    @property
    def supports_vision(self) -> bool:
        """Return whether this model supports vision"""
        return self._supports_vision

    async def _make_request(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Make async request to OpenAI API

        Args:
            messages: List of message objects
            max_tokens: Override max tokens
            temperature: Override temperature

        Returns:
            Generated text response
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature
        }

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"❌ OpenAI API error {response.status}: {error_text}")
                        raise Exception(f"OpenAI API error {response.status}: {error_text}")

                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]

                    logger.debug(f"✅ OpenAI API response received (length: {len(content)})")
                    return content

        except asyncio.TimeoutError:
            logger.error("❌ OpenAI API request timeout")
            raise Exception("OpenAI API request timeout")
        except Exception as e:
            logger.error(f"❌ OpenAI API request failed: {e}")
            raise

    def _encode_image(self, image_path: str) -> tuple:
        """
        Encode image to base64 for OpenAI API (with auto-resize if needed)

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
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return await self._make_request(messages, max_tokens, temperature)

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
        if not self.supports_vision:
            raise NotImplementedError(
                f"Model {self.model} does not support vision. "
                f"Use gpt-4o, gpt-4-vision-preview, or gpt-4-turbo instead."
            )

        try:
            # Encode image (with auto-resize if needed)
            image_base64, mime_type = self._encode_image(image_path)

            # Build messages
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # User message with image
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]

            messages.append({"role": "user", "content": user_content})

            response = await self._make_request(messages)

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
        if not self.supports_vision:
            raise NotImplementedError(
                f"Model {self.model} does not support vision. "
                f"Use gpt-4o, gpt-4-vision-preview, or gpt-4-turbo instead."
            )

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

        Args:
            image_paths: List of image file paths
            system_prompt: System prompt for OCR
            user_prompt: User prompt for OCR
            parallel: Whether to process images in parallel

        Returns:
            List of results with image path, extracted text, and status
        """
        if not self.supports_vision:
            raise NotImplementedError(
                f"Model {self.model} does not support vision. "
                f"Use gpt-4o, gpt-4-vision-preview, or gpt-4-turbo instead."
            )

        results = []

        if parallel:
            # Process all images concurrently
            tasks = []
            for image_path in image_paths:
                task = self._process_single_image_safe(image_path, system_prompt, user_prompt)
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
            for image_path in image_paths:
                try:
                    result = await self._process_single_image_safe(image_path, system_prompt, user_prompt)
                    results.append(result)
                except Exception as e:
                    results.append({
                        "image_path": image_path,
                        "text": "",
                        "success": False,
                        "error": str(e)
                    })

            return results

    async def _process_single_image_safe(
        self,
        image_path: str,
        system_prompt: str,
        user_prompt: str
    ) -> Dict[str, Any]:
        """
        Safely process a single image (with error handling)

        Returns:
            Result dictionary with success status
        """
        try:
            text = await self.analyze_image(image_path, user_prompt, system_prompt)
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
            # Combine texts into single prompt
            combined_text = "\n\n".join([f"[Text {i+1}]:\n{text}" for i, text in enumerate(texts)])

            # Build final prompt
            if user_prompt:
                final_prompt = f"{user_prompt}\n\n{combined_text}"
            else:
                final_prompt = f"Please synthesize the following texts into coherent content:\n\n{combined_text}"

            # Make request
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": final_prompt})

            response = await self._make_request(messages)

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
        Synthesize content from both images and texts (multimodal analysis)

        Args:
            image_paths: List of image file paths
            texts: List of texts (e.g., OCR results)
            system_prompt: System/context prompt
            user_prompt: User instructions for synthesis

        Returns:
            Synthesized content with visual and textual analysis
        """
        if not self.supports_vision:
            raise NotImplementedError(
                f"Model {self.model} does not support vision. "
                f"Use gpt-4o, gpt-4-vision-preview, or gpt-4-turbo instead."
            )

        try:
            logger.info(f"🔄 Synthesizing with {len(image_paths)} images + texts...")

            # Build content array with alternating images and texts
            content_blocks = []

            # Add all images with their texts
            for i, (image_path, text) in enumerate(zip(image_paths, texts), 1):
                # Add image (with auto-resize if needed)
                image_base64, mime_type = self._encode_image(image_path)

                content_blocks.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_base64}"
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

            # Build messages
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": content_blocks})

            # Make request
            response = await self._make_request(messages)

            logger.info(f"✅ Image+Text synthesis completed (output: {len(response)} characters)")
            return response

        except Exception as e:
            logger.error(f"❌ Image+Text synthesis failed: {e}")
            raise

    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information and statistics"""
        return {
            "provider": "ChatGPT (OpenAI)",
            "provider_type": self.provider_type.value,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "supports_vision": self.supports_vision,
            "status": "ready" if self.api_key else "missing_api_key"
        }
