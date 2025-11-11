"""
Request Validators

Shared validation logic for API requests to eliminate duplication.
"""

import os
from pathlib import Path
from typing import List, Optional
import logging

from fastapi import HTTPException
from core.exceptions import ValidationError, NotFoundError

logger = logging.getLogger(__name__)


class RequestValidator:
    """Shared request validation utilities"""

    @staticmethod
    def validate_folder_path(
        shared_uploads_path: str,
        folder_name: str,
        must_exist: bool = True
    ) -> Path:
        """
        Validate folder path and return Path object

        Args:
            shared_uploads_path: Base uploads directory
            folder_name: Folder name
            must_exist: Whether folder must exist

        Returns:
            Path object to folder

        Raises:
            HTTPException: If folder not found or invalid
        """
        try:
            folder_path = Path(shared_uploads_path) / folder_name

            # Check if path is safe (no directory traversal)
            folder_path = folder_path.resolve()
            base_path = Path(shared_uploads_path).resolve()

            if not str(folder_path).startswith(str(base_path)):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid folder path (directory traversal detected)"
                )

            # Check existence if required
            if must_exist and not folder_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Folder not found: {folder_path}"
                )

            # Check it's a directory
            if must_exist and not folder_path.is_dir():
                raise HTTPException(
                    status_code=400,
                    detail=f"Path is not a directory: {folder_path}"
                )

            return folder_path

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error validating folder path: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to validate folder path: {str(e)}"
            )

    @staticmethod
    def validate_output_format(
        format_name: str,
        available_formats: List[str],
        default: str = "default"
    ) -> str:
        """
        Validate and normalize output format

        Args:
            format_name: Requested format name
            available_formats: List of valid formats
            default: Default format to use if invalid

        Returns:
            Validated format name
        """
        if format_name not in available_formats:
            logger.warning(
                f"Unknown output format '{format_name}', using '{default}'"
            )
            return default
        return format_name

    @staticmethod
    def validate_game_code(game_code: str, min_length: int = 1) -> str:
        """
        Validate game code

        Args:
            game_code: Game code to validate
            min_length: Minimum length

        Returns:
            Validated game code

        Raises:
            ValidationError: If invalid
        """
        if not game_code or not game_code.strip():
            raise ValidationError("game_code cannot be empty", field="game_code")

        game_code = game_code.strip()

        if len(game_code) < min_length:
            raise ValidationError(
                f"game_code must be at least {min_length} characters",
                field="game_code",
                value=game_code
            )

        return game_code

    @staticmethod
    def validate_event_name(event_name: str, min_length: int = 1, max_length: int = 200) -> str:
        """
        Validate event name

        Args:
            event_name: Event name to validate
            min_length: Minimum length
            max_length: Maximum length

        Returns:
            Validated event name

        Raises:
            ValidationError: If invalid
        """
        if not event_name or not event_name.strip():
            raise ValidationError("event_name cannot be empty", field="event_name")

        event_name = event_name.strip()

        if len(event_name) < min_length:
            raise ValidationError(
                f"event_name must be at least {min_length} characters",
                field="event_name",
                value=event_name
            )

        if len(event_name) > max_length:
            raise ValidationError(
                f"event_name must not exceed {max_length} characters",
                field="event_name",
                value=f"{event_name[:50]}..."
            )

        return event_name

    @staticmethod
    def validate_positive_integer(
        value: int,
        field_name: str,
        min_value: int = 1,
        max_value: Optional[int] = None
    ) -> int:
        """
        Validate positive integer

        Args:
            value: Value to validate
            field_name: Field name for error messages
            min_value: Minimum allowed value
            max_value: Maximum allowed value (optional)

        Returns:
            Validated integer

        Raises:
            ValidationError: If invalid
        """
        if value < min_value:
            raise ValidationError(
                f"{field_name} must be at least {min_value}",
                field=field_name,
                value=value
            )

        if max_value is not None and value > max_value:
            raise ValidationError(
                f"{field_name} must not exceed {max_value}",
                field=field_name,
                value=value
            )

        return value

    @staticmethod
    def validate_text_content(
        text: str,
        field_name: str,
        min_length: int = 1,
        max_length: Optional[int] = None
    ) -> str:
        """
        Validate text content

        Args:
            text: Text to validate
            field_name: Field name for error messages
            min_length: Minimum length
            max_length: Maximum length (optional)

        Returns:
            Validated text

        Raises:
            ValidationError: If invalid
        """
        if not text or not text.strip():
            raise ValidationError(
                f"{field_name} cannot be empty",
                field=field_name
            )

        text = text.strip()

        if len(text) < min_length:
            raise ValidationError(
                f"{field_name} must be at least {min_length} characters",
                field=field_name,
                value=f"{text[:50]}..."
            )

        if max_length and len(text) > max_length:
            raise ValidationError(
                f"{field_name} must not exceed {max_length} characters",
                field=field_name,
                value=f"{text[:50]}..."
            )

        return text

    @staticmethod
    def validate_image_path(image_path: str) -> Path:
        """
        Validate image file path

        Args:
            image_path: Path to image file

        Returns:
            Path object

        Raises:
            ValidationError: If invalid
            NotFoundError: If file not found
        """
        path = Path(image_path)

        if not path.exists():
            raise NotFoundError("Image file", image_path)

        if not path.is_file():
            raise ValidationError(
                "Path is not a file",
                field="image_path",
                value=image_path
            )

        # Check file extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        if path.suffix.lower() not in valid_extensions:
            raise ValidationError(
                f"Unsupported image format: {path.suffix}",
                field="image_path",
                value=image_path,
                details={'valid_extensions': list(valid_extensions)}
            )

        return path

    @staticmethod
    def validate_file_size(
        file_path: Path,
        max_size_mb: int = 10
    ) -> int:
        """
        Validate file size

        Args:
            file_path: Path to file
            max_size_mb: Maximum size in megabytes

        Returns:
            File size in bytes

        Raises:
            ValidationError: If file too large
        """
        size_bytes = file_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        if size_mb > max_size_mb:
            raise ValidationError(
                f"File size ({size_mb:.1f} MB) exceeds maximum ({max_size_mb} MB)",
                field="file_size",
                value=f"{size_mb:.1f} MB",
                details={'max_size_mb': max_size_mb}
            )

        return size_bytes


class LogHelper:
    """Standardized logging helpers"""

    @staticmethod
    def log_request(endpoint: str, **kwargs):
        """Log incoming request with parameters"""
        logger.info(f"📥 Received request: {endpoint}")
        for key, value in kwargs.items():
            # Truncate long values
            if isinstance(value, str) and len(value) > 100:
                value = f"{value[:100]}..."
            logger.info(f"   - {key}: {value}")

    @staticmethod
    def log_success(operation: str, duration: Optional[float] = None, **kwargs):
        """Log successful operation"""
        msg = f"✅ {operation} completed"
        if duration:
            msg += f" in {duration:.2f}s"
        logger.info(msg)

        for key, value in kwargs.items():
            logger.info(f"   - {key}: {value}")

    @staticmethod
    def log_error(operation: str, error: Exception, **kwargs):
        """Log error with context"""
        logger.error(f"❌ {operation} failed: {error}")
        for key, value in kwargs.items():
            logger.error(f"   - {key}: {value}")

    @staticmethod
    def log_warning(message: str, **kwargs):
        """Log warning with context"""
        logger.warning(f"⚠️ {message}")
        for key, value in kwargs.items():
            logger.warning(f"   - {key}: {value}")


__all__ = ['RequestValidator', 'LogHelper']
