"""
Data Transfer Objects (DTOs)

Type-safe data structures to replace Dict[str, Any] throughout the application.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class EventTagsDTO:
    """Event categorization tags"""
    family: str
    dynamics: str
    rewards: str

    def to_dict(self) -> Dict[str, str]:
        return {
            'family': self.family,
            'dynamics': self.dynamics,
            'rewards': self.rewards
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'EventTagsDTO':
        return cls(
            family=data.get('family', ''),
            dynamics=data.get('dynamics', ''),
            rewards=data.get('rewards', '')
        )


@dataclass
class EventDTO:
    """Event data transfer object"""
    name: str
    about: str
    game_code: str
    tags: Optional[EventTagsDTO] = None
    faiss_index_name: Optional[int] = None
    faiss_index_about: Optional[int] = None
    faiss_index_images: Optional[List[int]] = None
    event_id: Optional[int] = None
    created_at: Optional[datetime] = None
    tag_explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'name': self.name,
            'about': self.about,
            'game_code': self.game_code,
        }

        if self.tags:
            result['tags'] = self.tags.to_dict()
        if self.faiss_index_name is not None:
            result['faiss_index_name'] = self.faiss_index_name
        if self.faiss_index_about is not None:
            result['faiss_index_about'] = self.faiss_index_about
        if self.faiss_index_images:
            result['faiss_index_images'] = self.faiss_index_images
        if self.event_id:
            result['id'] = self.event_id
        if self.created_at:
            result['created_at'] = self.created_at.isoformat()
        if self.tag_explanation:
            result['tag_explanation'] = self.tag_explanation

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventDTO':
        tags = None
        if 'tags' in data and data['tags']:
            tags = EventTagsDTO.from_dict(data['tags'])

        return cls(
            name=data.get('name', ''),
            about=data.get('about', ''),
            game_code=data.get('game_code', ''),
            tags=tags,
            faiss_index_name=data.get('faiss_index_name'),
            faiss_index_about=data.get('faiss_index_about'),
            faiss_index_images=data.get('faiss_index_images'),
            event_id=data.get('id') or data.get('event_id'),
            created_at=data.get('created_at'),
            tag_explanation=data.get('tag_explanation')
        )

    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> 'EventDTO':
        """Create DTO from database row"""
        return cls.from_dict(row)


@dataclass
class SimilarEventDTO:
    """Similar event with similarity scores"""
    event: EventDTO
    text_score: float
    image_score: float
    reason: str
    matching_images_count: int = 0
    image_faiss_indices: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.event.to_dict(),
            'score_text': self.text_score,
            'score_image': self.image_score,
            'reason': self.reason,
            'matching_images_count': self.matching_images_count,
            'image_faiss_indices': self.image_faiss_indices
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimilarEventDTO':
        # Extract event data (everything except similarity fields)
        event_data = {k: v for k, v in data.items()
                     if k not in ['score_text', 'score_image', 'reason',
                                 'matching_images_count', 'image_faiss_indices']}
        event = EventDTO.from_dict(event_data)

        return cls(
            event=event,
            text_score=data.get('score_text', 0.0),
            image_score=data.get('score_image', 0.0),
            reason=data.get('reason', ''),
            matching_images_count=data.get('matching_images_count', 0),
            image_faiss_indices=data.get('image_faiss_indices', [])
        )


@dataclass
class SearchResultDTO:
    """Search result with FAISS index and distance"""
    faiss_index: int
    distance: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'faiss_index': self.faiss_index,
            'distance': self.distance
        }

    @classmethod
    def from_tuple(cls, faiss_index: int, distance: float) -> 'SearchResultDTO':
        return cls(faiss_index=faiss_index, distance=distance)


@dataclass
class ImageEmbeddingDTO:
    """Image embedding result"""
    image_path: str
    faiss_index: int
    vector_dimension: int
    processing_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'image_path': self.image_path,
            'faiss_index': self.faiss_index,
            'vector_dimension': self.vector_dimension,
            'processing_time': self.processing_time
        }


@dataclass
class TextEmbeddingDTO:
    """Text embedding result"""
    text: str
    faiss_index: int
    vector_dimension: int
    processing_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text[:100],  # Truncate for logging
            'faiss_index': self.faiss_index,
            'vector_dimension': self.vector_dimension,
            'processing_time': self.processing_time
        }


@dataclass
class ServiceStatusDTO:
    """Service status information"""
    status: str  # "operational", "degraded", "unavailable"
    services: Dict[str, bool]
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'status': self.status,
            'services': self.services
        }
        if self.message:
            result['message'] = self.message
        if self.details:
            result['details'] = self.details
        return result

    @classmethod
    def operational(cls, services: Dict[str, bool],
                   details: Optional[Dict[str, Any]] = None) -> 'ServiceStatusDTO':
        return cls(
            status="operational",
            services=services,
            message="All services operational",
            details=details
        )

    @classmethod
    def degraded(cls, services: Dict[str, bool],
                message: str, details: Optional[Dict[str, Any]] = None) -> 'ServiceStatusDTO':
        return cls(
            status="degraded",
            services=services,
            message=message,
            details=details
        )

    @classmethod
    def unavailable(cls, services: Dict[str, bool],
                   message: str, details: Optional[Dict[str, Any]] = None) -> 'ServiceStatusDTO':
        return cls(
            status="unavailable",
            services=services,
            message=message,
            details=details
        )


@dataclass
class OCRResultDTO:
    """OCR extraction result"""
    image_path: str
    text: str
    success: bool
    error: Optional[str] = None
    processing_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'image_path': self.image_path,
            'text': self.text,
            'success': self.success
        }
        if self.error:
            result['error'] = self.error
        if self.processing_time:
            result['processing_time'] = self.processing_time
        return result


@dataclass
class AboutExtractionResultDTO:
    """About extraction result"""
    about_content: str
    processing_time: float
    image_count: int
    successful_ocr_count: int
    failed_ocr_count: int
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'about_content': self.about_content,
            'processing_time': self.processing_time,
            'image_count': self.image_count,
            'successful_ocr_count': self.successful_ocr_count,
            'failed_ocr_count': self.failed_ocr_count
        }
        if self.metadata:
            result['metadata'] = self.metadata
        return result


@dataclass
class SimilaritySearchRequestDTO:
    """Similarity search request parameters"""
    query_name: str
    query_about: str
    folder_name: str
    game_code: str
    shared_uploads_path: str
    image_count: int
    top_k: int = 10

    def validate(self) -> List[str]:
        """Validate request parameters, return list of errors"""
        errors = []

        if not self.query_name or not self.query_name.strip():
            errors.append("query_name cannot be empty")
        if not self.query_about or not self.query_about.strip():
            errors.append("query_about cannot be empty")
        if not self.folder_name or not self.folder_name.strip():
            errors.append("folder_name cannot be empty")
        if not self.game_code or not self.game_code.strip():
            errors.append("game_code cannot be empty")
        if self.image_count <= 0:
            errors.append("image_count must be positive")
        if self.top_k <= 0:
            errors.append("top_k must be positive")

        return errors


@dataclass
class SimilaritySearchResponseDTO:
    """Similarity search response"""
    query_event: EventDTO
    similar_events: List[SimilarEventDTO]
    processing_time: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'query_event': self.query_event.to_dict(),
            'similar_events': [e.to_dict() for e in self.similar_events],
            'processing_time': self.processing_time
        }
        if self.metadata:
            result['metadata'] = self.metadata
        return result


__all__ = [
    'EventTagsDTO',
    'EventDTO',
    'SimilarEventDTO',
    'SearchResultDTO',
    'ImageEmbeddingDTO',
    'TextEmbeddingDTO',
    'ServiceStatusDTO',
    'OCRResultDTO',
    'AboutExtractionResultDTO',
    'SimilaritySearchRequestDTO',
    'SimilaritySearchResponseDTO'
]
