"""
Claude Service - Handle all Claude API interactions for OCR and text synthesis
"""
import base64
import asyncio
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import aiohttp
import json
from config import config

logger = logging.getLogger(__name__)

class ClaudeService:
    def __init__(self):
        self.api_key = config.CLAUDE_API_KEY
        self.model = config.CLAUDE_MODEL
        self.base_url = "https://api.anthropic.com/v1"
        self.max_tokens = config.CLAUDE_MAX_TOKENS
        self.timeout = config.CLAUDE_TIMEOUT

        if not self.api_key:
            logger.warning("⚠️ CLAUDE_API_KEY not found in environment variables")
            logger.warning("⚠️ Claude service will not be functional until API key is provided")
            # Don't raise error, just warn
        else:
            logger.info(f"✅ Claude service initialized with model: {self.model}")
    
    async def _make_request_with_messages(self, messages: List[Dict], system_prompt: str = "") -> str:
        """
        Make async request to Claude API
        
        Args:
            messages: List of message objects
            system_prompt: System prompt for the conversation
            
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
            "max_tokens": self.max_tokens,
            "messages": messages
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
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

    async def _make_request(self, system_prompt: str = "", user_prompt: str = "", max_tokens: int = None) -> str:
        """
        Alternative _make_request method for simple text prompts
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt  
            max_tokens: Override max tokens
            
        Returns:
            Claude's response text
        """
        # Use custom max_tokens if provided
        if max_tokens:
            original_max_tokens = self.max_tokens
            self.max_tokens = max_tokens
        
        try:
            # Convert to messages format
            messages = [{
                "role": "user", 
                "content": user_prompt
            }]
            
            # Call existing _make_request method
            response = await self._make_request_with_messages(messages, system_prompt)
            
            return response
            
        finally:
            # Restore original max_tokens
            if max_tokens:
                self.max_tokens = original_max_tokens

    def _encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 for Claude API
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"❌ Error encoding image {image_path}: {e}")
            raise
    
    def _get_image_media_type(self, image_path: str) -> str:
        """
        Get media type for image based on file extension
        
        Args:
            image_path: Path to image file
            
        Returns:
            Media type string (e.g., 'image/png')
        """
        extension = Path(image_path).suffix.lower()
        media_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return media_types.get(extension, 'image/png')
    
    async def extract_text_from_image(self, 
                                    image_path: str, 
                                    system_prompt: str = "",
                                    user_prompt: str = "") -> str:
        """
        Extract text from image using Claude OCR
        
        Args:
            image_path: Path to image file
            system_prompt: System prompt for OCR
            user_prompt: User prompt for OCR
            
        Returns:
            Extracted text content
        """
        try:
            # Encode image
            image_base64 = self._encode_image(image_path)
            media_type = self._get_image_media_type(image_path)
            
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
                        "text": user_prompt or "Please extract all text from this image."
                    }
                ]
            }]
            
            # Make API request
            response = await self._make_request_with_messages(messages, system_prompt)
            
            logger.info(f"✅ OCR completed for {Path(image_path).name} (extracted {len(response)} characters)")
            return response
            
        except Exception as e:
            logger.error(f"❌ OCR failed for {image_path}: {e}")
            raise
    
    async def synthesize_content(self, 
                               ocr_texts: List[str],
                               system_prompt: str = "",
                               user_prompt: str = "") -> str:
        """
        Synthesize multiple OCR texts into coherent content
        
        Args:
            ocr_texts: List of OCR extracted texts
            system_prompt: System prompt for synthesis
            user_prompt: User prompt with placeholders filled
            
        Returns:
            Synthesized content
        """
        try:
            # Prepare messages
            messages = [{
                "role": "user",
                "content": user_prompt
            }]
            
            # Make API request
            response = await self._make_request_with_messages(messages, system_prompt)
            
            logger.info(f"✅ Content synthesis completed (output: {len(response)} characters)")
            return response
            
        except Exception as e:
            logger.error(f"❌ Content synthesis failed: {e}")
            raise
    
    async def process_multiple_images(self, 
                                    image_paths: List[str],
                                    system_prompt: str = "",
                                    user_prompt: str = "",
                                    parallel: bool = False) -> List[Dict[str, Any]]:
        """
        Process multiple images for OCR
        
        Args:
            image_paths: List of image file paths
            system_prompt: System prompt for OCR
            user_prompt: User prompt for OCR
            parallel: Whether to process images in parallel
            
        Returns:
            List of results with image path, extracted text, and status
        """
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
    
    async def _process_single_image_safe(self, 
                                       image_path: str, 
                                       system_prompt: str, 
                                       user_prompt: str) -> Dict[str, Any]:
        """
        Safely process a single image (with error handling)
        
        Returns:
            Result dictionary with success status
        """
        try:
            text = await self.extract_text_from_image(image_path, system_prompt, user_prompt)
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
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics (placeholder for future implementation)"""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "status": "ready"
        }

# Global instance - lazy loading
_claude_service_instance = None

def get_claude_service():
    """Get Claude service instance with lazy loading"""
    global _claude_service_instance
    if _claude_service_instance is None:
        _claude_service_instance = ClaudeService()
    return _claude_service_instance

# For backward compatibility
claude_service = None