"""
About Extraction Router - FastAPI endpoints for about content extraction
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
import os
from pathlib import Path

from services.about_extraction_service import about_extraction_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["about-extraction"])

# Request/Response Models
class ExtractAboutRequest(BaseModel):
    folder_name: str = Field(..., description="Name of the folder containing images")
    game_code: str = Field(default="", description="Game code or platform")
    event_name: str = Field(default="", description="Name of the event")
    event_type: str = Field(default="", description="Type of the event")
    image_count: int = Field(default=0, description="Expected number of images")
    shared_uploads_path: str = Field(..., description="Path to shared uploads directory")
    output_format: str = Field(default="default", description="Output format (default, json_simple, json_detailed, markdown, html)")
    process_parallel: bool = Field(default=True, description="Whether to process images in parallel")

class ExtractAboutResponse(BaseModel):
    success: bool
    about_content: Any = Field(description="Generated about content (format depends on output_format)")
    processing_time: float
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    family: Optional[str] = Field(None, description="Parsed mechanic family classification tag")
    dynamic: Optional[str] = Field(None, description="Parsed player dynamics classification tag")
    reward: Optional[str] = Field(None, description="Parsed reward types classification tag")

class ServiceStatusResponse(BaseModel):
    status: str
    claude_available: bool
    supported_formats: list
    supported_extensions: list
    version: str = "1.0.0"

@router.post("/extract-about", response_model=ExtractAboutResponse)
async def extract_about(request: ExtractAboutRequest) -> ExtractAboutResponse:
    """
    Extract about content from images in a folder using OCR + AI synthesis
    
    This endpoint:
    1. Finds all images in the specified folder
    2. Extracts text from each image using Claude OCR
    3. Synthesizes the extracted texts into coherent about content
    4. Returns the result in the specified format
    """
    try:
        logger.info(f"📥 Received extract-about request for folder: {request.folder_name}")
        
        # Construct full folder path
        folder_path = Path(request.shared_uploads_path) / request.folder_name
        
        # Validate folder exists
        if not folder_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Folder not found: {folder_path}"
            )
        
        # Log request details
        logger.info(f"🎯 Processing request:")
        logger.info(f"   - Folder: {folder_path}")
        logger.info(f"   - Event: {request.event_name} ({request.event_type})")
        logger.info(f"   - Game: {request.game_code}")
        logger.info(f"   - Output format: {request.output_format}")
        logger.info(f"   - Parallel processing: {request.process_parallel}")
        
        # Validate output format
        available_formats = about_extraction_service.get_supported_formats()
        if request.output_format not in available_formats:
            logger.warning(f"⚠️ Unknown output format '{request.output_format}', using 'default'")
            request.output_format = "default"
        
        # Process the request
        result = await about_extraction_service.extract_about_from_folder(
            folder_path=str(folder_path),
            event_name=request.event_name,
            event_type=request.event_type,
            game_code=request.game_code,
            output_format=request.output_format,
            process_parallel=request.process_parallel
        )
        
        # Log result summary
        if result["success"]:
            metadata = result.get("metadata", {})
            logger.info(f"✅ Successfully processed {metadata.get('successful_ocr_count', 0)} images")
            logger.info(f"   - Processing time: {result['processing_time']:.2f}s")
            logger.info(f"   - Content length: {len(str(result['about_content']))} characters")
        else:
            logger.error(f"❌ Processing failed: {result.get('error')}")
        
        # Return response with parsed tags
        return ExtractAboutResponse(
            success=result["success"],
            about_content=result["about_content"],
            processing_time=result["processing_time"],
            metadata=result.get("metadata"),
            error=result.get("error"),
            family=result.get("family"),
            dynamic=result.get("dynamic"),
            reward=result.get("reward")
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in extract-about endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/extract-about/status", response_model=ServiceStatusResponse)
async def get_service_status() -> ServiceStatusResponse:
    """
    Get the status of the about extraction service
    """
    try:
        service_status = about_extraction_service.get_service_status()
        
        return ServiceStatusResponse(
            status="healthy",
            claude_available=True,  # If we reach here, Claude service is likely working
            supported_formats=service_status["available_formats"],
            supported_extensions=service_status["supported_extensions"]
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting service status: {e}")
        return ServiceStatusResponse(
            status="error",
            claude_available=False,
            supported_formats=[],
            supported_extensions=[]
        )

@router.get("/extract-about/formats")
async def get_available_formats():
    """
    Get available output formats with their descriptions
    """
    try:
        formats = about_extraction_service.formatter.get_available_formats()
        format_info = {}
        
        for format_name in formats:
            info = about_extraction_service.formatter.get_format_info(format_name)
            format_info[format_name] = {
                "type": info.get("type", "unknown"),
                "description": info.get("description", "No description available")
            }
        
        return {
            "available_formats": format_info,
            "default_format": "default"
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting format info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving format information: {str(e)}"
        )

@router.post("/extract-about/reload-config")
async def reload_configuration():
    """
    Reload prompts and output format configurations
    """
    try:
        prompts_reloaded = about_extraction_service.prompt_manager.reload_prompts()
        formats_reloaded = about_extraction_service.formatter.reload_formats()
        
        return {
            "success": prompts_reloaded and formats_reloaded,
            "prompts_reloaded": prompts_reloaded,
            "formats_reloaded": formats_reloaded,
            "message": "Configuration reloaded successfully" if (prompts_reloaded and formats_reloaded) else "Some configurations failed to reload"
        }
        
    except Exception as e:
        logger.error(f"❌ Error reloading configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reloading configuration: {str(e)}"
        )

# Health check endpoint
@router.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "service": "about-extraction",
        "timestamp": "2025-01-01T00:00:00Z"  # You might want to use actual timestamp
    }