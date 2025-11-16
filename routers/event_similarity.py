"""
Event Similarity Analysis API Router

This module provides REST endpoints for comprehensive event alternative detection
using multi-modal search and taxonomy analysis.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

# Import the service (will be implemented)
# from services.event_similarity_service import EventSimilarityService

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(
    prefix="/api",
    tags=["find-similar-events"],
    responses={
        404: {"description": "Not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)

# Pydantic Models for Request/Response

class EventTags(BaseModel):
    """Event taxonomy tags"""
    family: str = Field(..., description="Event family taxonomy (e.g., 'Competitions', 'Challenges')")
    dynamics: str = Field(..., description="Event dynamics taxonomy (e.g., 'Collaborative', 'Individualistic')")
    rewards: str = Field(..., description="Event rewards taxonomy (e.g., 'Currencies & items', 'Items')")

class QueryEvent(BaseModel):
    """Query event information"""
    name: str = Field(..., description="Original event name")
    about: str = Field(..., description="Original event description")

class EventImage(BaseModel):
    """Event image information"""
    file_name: str = Field(..., description="Image filename")
    file_path: str = Field(..., description="Relative path to image file")
    created_at: Optional[str] = Field(None, description="Image creation timestamp (ISO format)")

class SimilarEvent(BaseModel):
    """Similar event with scores and analysis"""
    name: str = Field(..., description="Similar event name")
    about: str = Field(..., description="Similar event description")
    score_text: int = Field(..., ge=0, le=100, description="Text similarity score from Claude analysis (0-100)")
    score_image: int = Field(..., ge=0, le=100, description="Image similarity score normalized to 0-100")
    reason: str = Field(..., description="Explanation of why this event is considered similar (from Claude analysis if available)")
    images: List[EventImage] = Field(..., description="List of event images with file paths")

class FindSimilarEventsRequest(BaseModel):
    """Request model for finding similar events"""
    folder_name: str = Field(..., description="Name of the folder containing event images")
    game_code: str = Field(..., description="Game code identifier (e.g., 'candy_crush')")
    event_name: str = Field(..., description="Name of the event")
    about: str = Field(..., description="Event description/about content")
    image_count: int = Field(..., gt=0, description="Expected number of images for validation")
    shared_uploads_path: str = Field(default="/shared/uploads/", description="Base path for shared uploads")
    
    @validator('folder_name')
    def validate_folder_name(cls, v):
        """Validate folder name format"""
        if not v or not v.strip():
            raise ValueError("Folder name cannot be empty")
        # Remove any path separators for security
        clean_name = Path(v).name
        if not clean_name:
            raise ValueError("Invalid folder name")
        return clean_name
    
    @validator('game_code')
    def validate_game_code(cls, v):
        """Validate game code format"""
        if not v or not v.strip():
            raise ValueError("Game code cannot be empty")
        # Only strip whitespace, do NOT normalize to lowercase
        # The original game_code is needed for database queries
        return v.strip()
    
    @validator('shared_uploads_path')
    def validate_uploads_path(cls, v):
        """Validate uploads path format"""
        if not v or not v.strip():
            raise ValueError("Shared uploads path cannot be empty")
        # Ensure path ends with slash
        return v.rstrip('/') + '/'

class FindSimilarEventsResponse(BaseModel):
    """Response model for similar events analysis"""
    query_event: QueryEvent = Field(..., description="Information about the query event")
    similar_events: List[SimilarEvent] = Field(..., description="List of similar events with analysis")

class ServiceStatusResponse(BaseModel):
    """Service status response"""
    status: str = Field(..., description="Service status")
    database_connected: bool = Field(..., description="Database connection status")
    faiss_indexes_loaded: Dict[str, bool] = Field(..., description="FAISS index loading status by game")
    models_loaded: Dict[str, bool] = Field(..., description="AI models loading status")
    message: Optional[str] = Field(None, description="Additional status information")

class TaxonomyInfo(BaseModel):
    """Taxonomy information response"""
    family: List[str] = Field(..., description="Available family taxonomy values")
    dynamics: List[str] = Field(..., description="Available dynamics taxonomy values") 
    rewards: List[str] = Field(..., description="Available rewards taxonomy values")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

# This will be overridden by main.py dependency injection
async def get_event_similarity_service():
    """Dependency to get EventSimilarityService instance - will be overridden by main app"""
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="EventSimilarityService not initialized. Please ensure the service is properly configured."
    )

# API Endpoints

@router.post(
    "/find-similar-events",
    response_model=FindSimilarEventsResponse,
    summary="Find Similar Events",
    description="""
    Analyze an event with images to find similar alternative events using comprehensive multi-modal search.
    
    This endpoint:
    1. Loads images from the specified folder and validates count
    2. Performs multi-modal FAISS search (name, about, images with two-phase algorithm)
    3. Maps image scores fairly for all found events
    4. Analyzes all candidates with Claude for taxonomy tagging and alternative assessment
    5. Returns comprehensive analysis with separate text and image scores
    
    The two-phase image search ensures fair comparison between image-found and text-found events.
    """,
    responses={
        200: {
            "description": "Successfully analyzed similar events",
            "model": FindSimilarEventsResponse
        },
        400: {
            "description": "Invalid request parameters",
            "model": ErrorResponse
        },
        404: {
            "description": "Images folder not found or empty",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal processing error",
            "model": ErrorResponse
        }
    }
)
async def find_similar_events(
    request: FindSimilarEventsRequest,
    service = Depends(get_event_similarity_service)
):
    """
    Find similar events using comprehensive multi-modal analysis
    """
    try:
        logger.info(f"Processing similarity request for event: {request.event_name}")
        logger.info(f"Game: {request.game_code}, Folder: {request.folder_name}, Images: {request.image_count}")
        
        # Call the service to perform the analysis
        result = await service.find_similar_events(
            folder_name=request.folder_name,
            game_code=request.game_code,
            event_name=request.event_name,
            about=request.about,
            image_count=request.image_count,
            shared_uploads_path=request.shared_uploads_path
        )
        
        logger.info(f"Found {len(result['similar_events'])} similar events")
        return FindSimilarEventsResponse(**result)
        
    except FileNotFoundError as e:
        logger.error(f"Images folder not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Images folder not found: {str(e)}"
        )
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in find_similar_events: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal processing error: {str(e)}"
        )

@router.get(
    "/find-similar-events/status",
    response_model=ServiceStatusResponse,
    summary="Get Service Status",
    description="Check the health status of the event similarity analysis service including database and AI models."
)
async def get_service_status(
    service = Depends(get_event_similarity_service)
):
    """
    Get comprehensive service status including dependencies
    """
    try:
        logger.info("Checking service status")
        
        status_info = await service.get_service_status()
        
        return ServiceStatusResponse(**status_info)
        
    except Exception as e:
        logger.error(f"Error getting service status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking service status: {str(e)}"
        )

@router.get(
    "/find-similar-events/taxonomy",
    response_model=TaxonomyInfo,
    summary="Get Available Taxonomy Values",
    description="Retrieve all available taxonomy values for event classification (family, dynamics, rewards)."
)
async def get_taxonomy_info(
    service = Depends(get_event_similarity_service)
):
    """
    Get available taxonomy values for event classification
    """
    try:
        logger.info("Retrieving taxonomy information")
        
        # Static taxonomy values based on the configuration
        taxonomy_info = {
            "family": [
                "Accelerators", "Banks", "Challenges", "Clubs", "Collections", 
                "Competitions", "Custom Design", "Expansions", "Hazards", 
                "Interactions", "Levels", "Mini-Games", "Missions", "Notices", 
                "Other", "Purchases", "Quests", "Rewards"
            ],
            "dynamics": [
                "Collaborative", "Individualistic", "Collaborative & competitive",
                "Individualistic & competitive", "Collaborative & individualistic", 
                "Indiv. collab. & comp"
            ],
            "rewards": [
                "Currencies", "Currencies & items", "Currencies & real prize",
                "Currencies items & real prize", "Items", "Items & real prize", 
                "None", "Real prize"
            ]
        }
        
        return TaxonomyInfo(**taxonomy_info)
        
    except Exception as e:
        logger.error(f"Error getting taxonomy info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving taxonomy information: {str(e)}"
        )

@router.post(
    "/find-similar-events/validate",
    summary="Validate Request Parameters",
    description="Validate request parameters without performing the full analysis. Useful for checking folder existence and parameter validity.",
    responses={
        200: {"description": "Parameters are valid"},
        400: {"description": "Invalid parameters"},
        404: {"description": "Folder not found"}
    }
)
async def validate_request(
    request: FindSimilarEventsRequest,
    service = Depends(get_event_similarity_service)
):
    """
    Validate request parameters without performing full analysis
    """
    try:
        logger.info(f"Validating request for folder: {request.folder_name}")
        
        # Construct full folder path
        folder_path = Path(request.shared_uploads_path) / request.folder_name
        
        # Check if folder exists
        if not folder_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Folder not found: {folder_path}"
            )
        
        # Check if folder contains images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [
            f for f in folder_path.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if len(image_files) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No image files found in the specified folder"
            )
        
        if len(image_files) != request.image_count:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image count mismatch: expected {request.image_count}, found {len(image_files)}"
            )
        
        return {
            "status": "valid",
            "message": f"Found {len(image_files)} images in folder {request.folder_name}",
            "folder_path": str(folder_path),
            "image_files": [f.name for f in image_files]
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error validating request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation error: {str(e)}"
        )