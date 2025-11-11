"""
Custom Exception Classes

Provides a hierarchy of exceptions for better error handling and debugging.
"""

from typing import Optional, Dict, Any


class ImageSimilarityError(Exception):
    """Base exception for all application errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'details': self.details
        }


class ValidationError(ImageSimilarityError):
    """Raised when input validation fails"""

    def __init__(self, message: str, field: Optional[str] = None,
                 value: Optional[Any] = None, details: Optional[Dict[str, Any]] = None):
        self.field = field
        self.value = value
        details = details or {}
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = str(value)
        super().__init__(message, details)


class ServiceUnavailableError(ImageSimilarityError):
    """Raised when an external service is unavailable"""

    def __init__(self, service_name: str, message: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        self.service_name = service_name
        details = details or {}
        details['service'] = service_name
        message = message or f"Service '{service_name}' is currently unavailable"
        super().__init__(message, details)


class DatabaseError(ImageSimilarityError):
    """Raised when database operations fail"""

    def __init__(self, message: str, operation: Optional[str] = None,
                 query: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.operation = operation
        self.query = query
        details = details or {}
        if operation:
            details['operation'] = operation
        if query:
            details['query'] = query
        super().__init__(message, details)


class ModelError(ImageSimilarityError):
    """Raised when ML model operations fail"""

    def __init__(self, model_name: str, message: str,
                 details: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        details = details or {}
        details['model'] = model_name
        super().__init__(message, details)


class ConfigurationError(ImageSimilarityError):
    """Raised when configuration is invalid or missing"""

    def __init__(self, message: str, config_key: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        self.config_key = config_key
        details = details or {}
        if config_key:
            details['config_key'] = config_key
        super().__init__(message, details)


class NotFoundError(ImageSimilarityError):
    """Raised when a resource is not found"""

    def __init__(self, resource_type: str, resource_id: Optional[str] = None,
                 message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.resource_type = resource_type
        self.resource_id = resource_id
        details = details or {}
        details['resource_type'] = resource_type
        if resource_id:
            details['resource_id'] = resource_id
        message = message or f"{resource_type} not found"
        if resource_id:
            message += f": {resource_id}"
        super().__init__(message, details)


class FAISSIndexError(ImageSimilarityError):
    """Raised when FAISS index operations fail"""

    def __init__(self, message: str, game_code: Optional[str] = None,
                 content_type: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.game_code = game_code
        self.content_type = content_type
        details = details or {}
        if game_code:
            details['game_code'] = game_code
        if content_type:
            details['content_type'] = content_type
        super().__init__(message, details)


class ImageProcessingError(ImageSimilarityError):
    """Raised when image processing fails"""

    def __init__(self, message: str, image_path: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        self.image_path = image_path
        details = details or {}
        if image_path:
            details['image_path'] = image_path
        super().__init__(message, details)


class TextProcessingError(ImageSimilarityError):
    """Raised when text processing fails"""

    def __init__(self, message: str, text_snippet: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        self.text_snippet = text_snippet
        details = details or {}
        if text_snippet:
            # Include only first 100 chars for debugging
            details['text_snippet'] = text_snippet[:100] if text_snippet else None
        super().__init__(message, details)


# Export all exceptions
__all__ = [
    'ImageSimilarityError',
    'ValidationError',
    'ServiceUnavailableError',
    'DatabaseError',
    'ModelError',
    'ConfigurationError',
    'NotFoundError',
    'FAISSIndexError',
    'ImageProcessingError',
    'TextProcessingError'
]
