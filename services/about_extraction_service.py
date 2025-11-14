"""
About Extraction Service - Main service for extracting and synthesizing about content from images
"""
import os
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from .claude_service import get_claude_service
from utils.prompt_manager import PromptManager
from utils.output_formatter import output_formatter
from utils.tag_parser import get_tag_parser

logger = logging.getLogger(__name__)

class AboutExtractionService:
    def __init__(self):
        self.claude = None  # Will be lazy loaded
        self.prompt_manager = PromptManager() 
        self.formatter = output_formatter
        
        # Supported image extensions
        self.supported_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
        
        logger.info("✅ About Extraction Service initialized")
    
    def _get_claude_service(self):
        """Lazy load Claude service"""
        if self.claude is None:
            self.claude = get_claude_service()
        return self.claude
    
    async def extract_about_from_folder(self,
                                      folder_path: str,
                                      event_name: str = "",
                                      event_type: str = "",
                                      game_code: str = "",
                                      output_format: str = "default",
                                      process_parallel: bool = False) -> Dict[str, Any]:
        """
        Main method to extract about content from a folder of images
        
        Args:
            folder_path: Path to folder containing images
            event_name: Name of the event
            event_type: Type of the event
            game_code: Game code/platform
            output_format: Output format name (from config)
            process_parallel: Whether to process images in parallel
            
        Returns:
            Formatted result containing about content and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Starting about extraction for folder: {folder_path}")
            
            # Step 1: Find images in folder
            image_paths = self._find_images_in_folder(folder_path)
            if not image_paths:
                raise ValueError(f"No supported images found in folder: {folder_path}")
            
            logger.info(f"📸 Found {len(image_paths)} images to process")
            
            # Step 2: Extract text from all images using OCR
            ocr_results = await self._extract_texts_from_images(image_paths, process_parallel)
            
            # Filter successful OCR results
            successful_ocrs = [result for result in ocr_results if result["success"]]
            failed_ocrs = [result for result in ocr_results if not result["success"]]
            
            if failed_ocrs:
                logger.warning(f"⚠️ {len(failed_ocrs)} images failed OCR processing")
                for failed in failed_ocrs:
                    logger.warning(f"   - {Path(failed['image_path']).name}: {failed['error']}")
            
            if not successful_ocrs:
                raise Exception("All OCR attempts failed")
            
            logger.info(f"✅ Successfully processed {len(successful_ocrs)}/{len(image_paths)} images")
            
            # Step 3: Synthesize content from OCR texts
            about_content = await self._synthesize_about_content(
                successful_ocrs, event_name, event_type, game_code
            )

            # Step 3.5: Parse classification tags from content
            parsed_tags = self._parse_classification_tags(about_content)

            # Step 4: Prepare metadata
            processing_time = time.time() - start_time
            metadata = {
                "event_name": event_name,
                "event_type": event_type,
                "game_code": game_code,
                "image_count": len(image_paths),
                "successful_ocr_count": len(successful_ocrs),
                "failed_ocr_count": len(failed_ocrs),
                "processing_time": round(processing_time, 2),
                "ocr_results": ocr_results,
                "folder_path": folder_path
            }
            
            # Step 5: Format output
            formatted_result = self.formatter.format_output(
                content=about_content,
                format_name=output_format,
                metadata=metadata
            )
            
            logger.info(f"🎉 About extraction completed in {processing_time:.2f}s")

            return {
                "success": True,
                "about_content": about_content if output_format == "default" else formatted_result,
                "metadata": metadata,
                "processing_time": processing_time,
                "family": parsed_tags.get("family"),
                "dynamic": parsed_tags.get("dynamic"),
                "reward": parsed_tags.get("reward")
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ About extraction failed after {processing_time:.2f}s: {e}")

            return {
                "success": False,
                "error": str(e),
                "about_content": "",
                "processing_time": processing_time,
                "metadata": {
                    "processing_time": processing_time,
                    "folder_path": folder_path
                }
            }
    
    def _find_images_in_folder(self, folder_path: str) -> List[str]:
        """
        Find all supported image files in the folder
        
        Args:
            folder_path: Path to search for images
            
        Returns:
            List of image file paths
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        image_paths = []
        for file_path in folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                image_paths.append(str(file_path))
        
        # Sort by filename for consistent processing order
        image_paths.sort()
        
        logger.debug(f"📁 Found images: {[Path(p).name for p in image_paths]}")
        return image_paths
    
    async def _extract_texts_from_images(self, 
                                       image_paths: List[str], 
                                       parallel: bool = False) -> List[Dict[str, Any]]:
        """
        Extract text from multiple images using OCR
        
        Args:
            image_paths: List of image file paths
            parallel: Whether to process in parallel
            
        Returns:
            List of OCR results
        """
        # Get OCR prompts
        ocr_prompts = self.prompt_manager.get_ocr_prompts()
        
        logger.info(f"🔍 Starting OCR for {len(image_paths)} images (parallel: {parallel})")
        
        # Process images
        results = await self._get_claude_service().process_multiple_images(
            image_paths=image_paths,
            system_prompt=ocr_prompts["system"],
            user_prompt=ocr_prompts["user"],
            parallel=parallel
        )
        
        return results
    
    async def _synthesize_about_content(self,
                                      ocr_results: List[Dict[str, Any]],
                                      event_name: str,
                                      event_type: str,
                                      game_code: str) -> str:
        """
        Synthesize about content from OCR results (v2.0 workflow)

        This method uses the new workflow that sends both images and OCR texts
        to Claude, allowing it to perform visual analysis for classification.

        Args:
            ocr_results: List of successful OCR results
            event_name: Event name
            event_type: Event type
            game_code: Game code

        Returns:
            Synthesized bilingual about content with classification tags

        Raises:
            Exception: If no valid OCR text found or synthesis fails
        """
        # Prepare OCR texts for synthesis
        ocr_texts_formatted = []
        for i, result in enumerate(ocr_results, 1):
            image_name = Path(result["image_path"]).name
            text = result["text"].strip()
            if text:
                ocr_texts_formatted.append(f"Image {i} ({image_name}):\n{text}")

        if not ocr_texts_formatted:
            raise Exception("No valid OCR text found for synthesis")

        # Combine all OCR texts
        combined_ocr_text = "\n\n---\n\n".join(ocr_texts_formatted)

        # Get synthesis prompts with substitution
        synthesis_prompts = self.prompt_manager.get_synthesis_prompts(
            event_name=event_name,
            event_type=event_type,
            game_code=game_code,
            image_count=len(ocr_results),
            ocr_texts=combined_ocr_text
        )

        logger.info(f"🔄 Synthesizing content from {len(ocr_results)} OCR results (v2.0 with images)")

        # Extract image paths and texts
        image_paths = [result["image_path"] for result in ocr_results]
        ocr_texts = [result["text"] for result in ocr_results]

        # Use new v2.0 workflow: Send both images and OCR texts
        about_content = await self._get_claude_service().synthesize_with_images_and_texts(
            image_paths=image_paths,
            ocr_texts=ocr_texts,
            system_prompt=synthesis_prompts["system"],
            user_prompt=synthesis_prompts["user"]
        )

        return about_content.strip()

    def _parse_classification_tags(self, about_content: str) -> Dict[str, Optional[str]]:
        """
        Parse classification tags from about content

        Args:
            about_content: Full about content with critique section

        Returns:
            Dictionary with 'family', 'dynamic', 'reward' keys
        """
        try:
            logger.info("🔍 Parsing classification tags from about content...")
            parser = get_tag_parser()
            tags = parser.parse_tags(about_content)

            # Log parsed tags summary
            parsed_count = sum(1 for v in tags.values() if v is not None)
            logger.info(f"✅ Parsed {parsed_count}/3 tags successfully")

            return tags
        except Exception as e:
            logger.error(f"❌ Failed to parse tags: {e}", exc_info=True)
            return {
                "family": None,
                "dynamic": None,
                "reward": None
            }

    def get_supported_formats(self) -> List[str]:
        """Get list of supported output formats"""
        return self.formatter.get_available_formats()
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status and configuration"""
        try:
            claude_status = self._get_claude_service().get_usage_stats()
        except:
            claude_status = {"status": "unavailable"}
            
        return {
            "claude_status": claude_status,
            "supported_extensions": list(self.supported_extensions),
            "available_formats": self.get_supported_formats(),
            "prompt_categories": self.prompt_manager.get_available_categories()
        }

# Global instance
about_extraction_service = AboutExtractionService()

# Convenience function
async def extract_about_from_folder(**kwargs) -> Dict[str, Any]:
    """Convenience function for about extraction"""
    return await about_extraction_service.extract_about_from_folder(**kwargs)