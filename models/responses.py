"""
API Response Models

Standardized response structures for all API endpoints.
"""

from typing import Optional, Any, Dict, Generic, TypeVar, List
from datetime import datetime
from pydantic import BaseModel, Field

T = TypeVar('T')


class ErrorDetail(BaseModel):
    """Error details"""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    field: Optional[str] = Field(None, description="Field that caused the error")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class ResponseMetadata(BaseModel):
    """Response metadata"""
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    version: str = Field(default="2.0.0", description="API version")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class APIResponse(BaseModel, Generic[T]):
    """Generic API response wrapper"""
    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[T] = Field(None, description="Response data")
    error: Optional[ErrorDetail] = Field(None, description="Error details if failed")
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata, description="Response metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {"result": "example"},
                "error": None,
                "metadata": {
                    "timestamp": "2025-01-01T00:00:00",
                    "processing_time": 1.23,
                    "version": "2.0.0"
                }
            }
        }


class PaginationInfo(BaseModel):
    """Pagination information"""
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, le=100, description="Items per page")
    total_items: int = Field(..., ge=0, description="Total number of items")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated API response"""
    success: bool = Field(..., description="Whether the request was successful")
    data: List[T] = Field(..., description="List of items")
    pagination: PaginationInfo = Field(..., description="Pagination information")
    error: Optional[ErrorDetail] = Field(None, description="Error details if failed")
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Health status: healthy, degraded, unhealthy")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(default="2.0.0")
    services: Optional[Dict[str, bool]] = Field(None, description="Service health status")
    uptime: Optional[float] = Field(None, description="Uptime in seconds")


class SuccessResponse(BaseModel):
    """Simple success response"""
    success: bool = Field(default=True)
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data")


class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = Field(default=False)
    error: ErrorDetail = Field(..., description="Error details")
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


# Helper functions for creating responses
def success_response(
    data: Any,
    processing_time: Optional[float] = None,
    request_id: Optional[str] = None
) -> APIResponse:
    """Create a success response"""
    return APIResponse(
        success=True,
        data=data,
        error=None,
        metadata=ResponseMetadata(
            processing_time=processing_time,
            request_id=request_id
        )
    )


def error_response(
    code: str,
    message: str,
    field: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    processing_time: Optional[float] = None
) -> APIResponse:
    """Create an error response"""
    return APIResponse(
        success=False,
        data=None,
        error=ErrorDetail(
            code=code,
            message=message,
            field=field,
            details=details
        ),
        metadata=ResponseMetadata(processing_time=processing_time)
    )


def paginated_response(
    items: List[Any],
    page: int,
    page_size: int,
    total_items: int,
    processing_time: Optional[float] = None
) -> PaginatedResponse:
    """Create a paginated response"""
    total_pages = (total_items + page_size - 1) // page_size

    return PaginatedResponse(
        success=True,
        data=items,
        pagination=PaginationInfo(
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        ),
        error=None,
        metadata=ResponseMetadata(processing_time=processing_time)
    )


__all__ = [
    'APIResponse',
    'PaginatedResponse',
    'HealthResponse',
    'SuccessResponse',
    'ErrorResponse',
    'ErrorDetail',
    'ResponseMetadata',
    'PaginationInfo',
    'success_response',
    'error_response',
    'paginated_response'
]
