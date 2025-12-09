"""
Determine Alternative API Router

This module provides REST endpoint for determining if a new event is an alternative
of existing candidate events using LLM analysis.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(
    prefix="/api",
    tags=["determine-alternative"],
    responses={
        400: {"description": "Bad request"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)


# ==================== Request Models ====================

class NewEventInput(BaseModel):
    """New event information from user input"""
    name: str = Field(..., min_length=1, description="Name of the new event")
    about: str = Field(..., min_length=1, description="Description/about content of the new event")


class CandidateEventInput(BaseModel):
    """Candidate event information from Node.js (fetched from DB)"""
    code: str = Field(..., min_length=1, description="Event code UUID")
    name: str = Field(..., min_length=1, description="Event name")
    about: str = Field(..., description="Event about/description text")


class DetermineAlternativeRequest(BaseModel):
    """Request model for determine alternative endpoint"""
    game_code: str = Field(..., min_length=1, description="Game code identifier")
    new_event: NewEventInput = Field(..., description="New event to analyze")
    candidate_events: List[CandidateEventInput] = Field(
        ...,
        min_items=2,
        max_items=2,
        description="Exactly 2 candidate events to compare against"
    )

    @validator('candidate_events')
    def validate_candidate_count(cls, v):
        """Ensure exactly 2 candidate events"""
        if len(v) != 2:
            raise ValueError("Exactly 2 candidate events are required")
        return v


# ==================== Response Models ====================

class EventImage(BaseModel):
    """Event image information"""
    file_name: str = Field(..., description="Image filename")
    file_path: str = Field(..., description="Path to image file")


class AlternativeResult(BaseModel):
    """Result for each candidate event"""
    event_code: str = Field(..., description="Candidate event code UUID")
    event_name: str = Field(..., description="Candidate event name")
    is_alternative: bool = Field(..., description="Whether new event is an alternative of this candidate")
    score: int = Field(..., ge=0, le=100, description="Similarity score (0-100)")
    change_types: List[str] = Field(
        default=[],
        description="List of change types: None, Redesign, Rename, Reprice, Restructure (can have multiple)"
    )
    reason: str = Field(..., description="Explanation from LLM (in Vietnamese)")
    images: List[EventImage] = Field(default=[], description="List of event images")


class NewEventOutput(BaseModel):
    """New event info in response"""
    name: str = Field(..., description="New event name")
    about: str = Field(..., description="New event about")


class DetermineAlternativeResponse(BaseModel):
    """Response model for determine alternative endpoint"""
    success: bool = Field(..., description="Whether the request was successful")
    new_event: NewEventOutput = Field(..., description="New event information")
    alternatives: List[AlternativeResult] = Field(..., description="Analysis results for each candidate")


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(default=False)
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")


# ==================== Dependency ====================

async def get_determine_alternative_service():
    """Dependency to get DetermineAlternativeService instance - will be overridden by main app"""
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="DetermineAlternativeService not initialized"
    )


# ==================== Endpoint ====================

@router.post(
    "/determine-alternative",
    response_model=DetermineAlternativeResponse,
    summary="Determine Alternative Events",
    description="""
    Analyze whether a new event is an alternative (similar/repeated version) of existing candidate events.

    This endpoint:
    1. Receives a new event and 2 candidate events from Node.js
    2. Sends all events to LLM for comparison analysis
    3. Returns similarity scores and explanations for each candidate
    4. Fetches images for each candidate event from database
    """,
    responses={
        200: {
            "description": "Successfully analyzed alternatives",
            "model": DetermineAlternativeResponse
        },
        400: {
            "description": "Invalid request parameters",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal processing error",
            "model": ErrorResponse
        }
    }
)
async def determine_alternative(
    request: DetermineAlternativeRequest,
    service=Depends(get_determine_alternative_service)
):
    """
    Determine if a new event is an alternative of candidate events
    """
    try:
        logger.info(f"🔍 Processing determine alternative request")
        logger.info(f"   Game: {request.game_code}")
        logger.info(f"   New event: {request.new_event.name}")
        logger.info(f"   Candidates: {[c.name for c in request.candidate_events]}")

        # Call service to perform analysis
        result = await service.determine_alternative(
            game_code=request.game_code,
            new_event={
                "name": request.new_event.name,
                "about": request.new_event.about
            },
            candidate_events=[
                {
                    "code": c.code,
                    "name": c.name,
                    "about": c.about
                }
                for c in request.candidate_events
            ]
        )

        logger.info(f"✅ Analysis complete")
        return DetermineAlternativeResponse(**result)

    except ValueError as e:
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"❌ Error in determine_alternative: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal processing error: {str(e)}"
        )
