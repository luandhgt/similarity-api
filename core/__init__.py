"""
Core Module

Contains core functionality like exceptions, base classes, and shared utilities.
"""

from .exceptions import (
    ImageSimilarityError,
    ValidationError,
    ServiceUnavailableError,
    DatabaseError,
    ModelError,
    ConfigurationError,
    NotFoundError
)

__all__ = [
    'ImageSimilarityError',
    'ValidationError',
    'ServiceUnavailableError',
    'DatabaseError',
    'ModelError',
    'ConfigurationError',
    'NotFoundError'
]
