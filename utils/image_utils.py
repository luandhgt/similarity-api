"""
Image Utilities

Consolidates image-related operations to eliminate duplication.
"""

import logging
from pathlib import Path
from typing import List, Set, Optional
import glob

from core.exceptions import NotFoundError, ValidationError, ImageProcessingError

logger = logging.getLogger(__name__)


class ImageUtils:
    """Unified image utilities"""

    # Supported image extensions
    SUPPORTED_EXTENSIONS: Set[str] = {
        '.jpg', '.jpeg', '.png', '.bmp',
        '.tiff', '.tif', '.webp', '.gif'
    }

    @staticmethod
    def find_images_in_folder(
        folder_path: str,
        expected_count: Optional[int] = None,
        recursive: bool = False
    ) -> List[Path]:
        """
        Find all supported images in folder

        Args:
            folder_path: Path to folder
            expected_count: Expected number of images (logs warning if different)
            recursive: Search recursively in subdirectories

        Returns:
            List of Path objects to image files

        Raises:
            NotFoundError: If folder not found
            ValidationError: If no images found
        """
        folder = Path(folder_path)

        if not folder.exists():
            raise NotFoundError("Folder", str(folder_path))

        if not folder.is_dir():
            raise ValidationError(
                "Path is not a directory",
                field="folder_path",
                value=str(folder_path)
            )

        # Find images
        image_files = []

        for ext in ImageUtils.SUPPORTED_EXTENSIONS:
            # Search for both lowercase and uppercase extensions
            if recursive:
                pattern = f"**/*{ext}"
            else:
                pattern = f"*{ext}"

            image_files.extend(folder.glob(pattern))

            # Also search uppercase
            if recursive:
                pattern_upper = f"**/*{ext.upper()}"
            else:
                pattern_upper = f"*{ext.upper()}"

            image_files.extend(folder.glob(pattern_upper))

        # Remove duplicates and sort
        image_files = sorted(set(image_files))

        # Validate found images
        if not image_files:
            raise ValidationError(
                f"No supported images found in folder",
                field="folder_path",
                value=str(folder_path),
                details={'supported_extensions': list(ImageUtils.SUPPORTED_EXTENSIONS)}
            )

        # Check expected count
        if expected_count is not None and len(image_files) != expected_count:
            logger.warning(
                f"Expected {expected_count} images, found {len(image_files)} in {folder_path}"
            )
            logger.warning(f"Found images: {[f.name for f in image_files]}")

        logger.info(f"Found {len(image_files)} images in {folder_path}")

        return image_files

    @staticmethod
    def find_images_using_glob(
        folder_path: str,
        expected_count: Optional[int] = None
    ) -> List[str]:
        """
        Find images using glob patterns (alternative implementation)

        Args:
            folder_path: Path to folder
            expected_count: Expected number of images

        Returns:
            List of image file paths as strings

        Raises:
            NotFoundError: If folder not found
            ValidationError: If no images found
        """
        if not Path(folder_path).exists():
            raise NotFoundError("Folder", folder_path)

        # Build glob patterns
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []

        for ext in image_extensions:
            # Lowercase
            image_files.extend(glob.glob(str(Path(folder_path) / ext)))
            # Uppercase
            image_files.extend(glob.glob(str(Path(folder_path) / ext.upper())))

        # Remove duplicates and sort
        image_files = sorted(set(image_files))

        if not image_files:
            raise ValidationError(
                f"No supported images found in folder",
                field="folder_path",
                value=folder_path
            )

        # Check expected count
        if expected_count is not None and len(image_files) != expected_count:
            logger.warning(
                f"Expected {expected_count} images, found {len(image_files)} in {folder_path}"
            )

        return image_files

    @staticmethod
    def validate_image_file(image_path: str) -> Path:
        """
        Validate image file exists and is supported

        Args:
            image_path: Path to image

        Returns:
            Path object

        Raises:
            NotFoundError: If file not found
            ValidationError: If unsupported format
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

        # Check extension
        if path.suffix.lower() not in ImageUtils.SUPPORTED_EXTENSIONS:
            raise ValidationError(
                f"Unsupported image format: {path.suffix}",
                field="image_path",
                value=image_path,
                details={'supported_extensions': list(ImageUtils.SUPPORTED_EXTENSIONS)}
            )

        return path

    @staticmethod
    def get_image_info(image_path: Path) -> dict:
        """
        Get image file information

        Args:
            image_path: Path to image

        Returns:
            Dictionary with image info
        """
        stat = image_path.stat()
        return {
            'name': image_path.name,
            'path': str(image_path),
            'extension': image_path.suffix.lower(),
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'modified_time': stat.st_mtime
        }

    @staticmethod
    def filter_by_size(
        image_paths: List[Path],
        min_size_kb: Optional[int] = None,
        max_size_kb: Optional[int] = None
    ) -> List[Path]:
        """
        Filter images by file size

        Args:
            image_paths: List of image paths
            min_size_kb: Minimum size in KB
            max_size_kb: Maximum size in KB

        Returns:
            Filtered list of image paths
        """
        filtered = []

        for path in image_paths:
            size_kb = path.stat().st_size / 1024

            if min_size_kb and size_kb < min_size_kb:
                logger.debug(f"Skipping {path.name}: too small ({size_kb:.1f} KB)")
                continue

            if max_size_kb and size_kb > max_size_kb:
                logger.debug(f"Skipping {path.name}: too large ({size_kb:.1f} KB)")
                continue

            filtered.append(path)

        return filtered

    @staticmethod
    def sort_images(
        image_paths: List[Path],
        sort_by: str = 'name'
    ) -> List[Path]:
        """
        Sort images by various criteria

        Args:
            image_paths: List of image paths
            sort_by: Sort criterion ('name', 'size', 'modified')

        Returns:
            Sorted list of image paths
        """
        if sort_by == 'name':
            return sorted(image_paths, key=lambda p: p.name)
        elif sort_by == 'size':
            return sorted(image_paths, key=lambda p: p.stat().st_size)
        elif sort_by == 'modified':
            return sorted(image_paths, key=lambda p: p.stat().st_mtime)
        else:
            logger.warning(f"Unknown sort criterion '{sort_by}', sorting by name")
            return sorted(image_paths, key=lambda p: p.name)

    @staticmethod
    def get_supported_extensions() -> Set[str]:
        """Get set of supported image extensions"""
        return ImageUtils.SUPPORTED_EXTENSIONS.copy()

    @staticmethod
    def is_supported_format(file_path: str) -> bool:
        """
        Check if file format is supported

        Args:
            file_path: Path to file

        Returns:
            True if supported
        """
        ext = Path(file_path).suffix.lower()
        return ext in ImageUtils.SUPPORTED_EXTENSIONS

    @staticmethod
    def count_images_in_folder(folder_path: str, recursive: bool = False) -> int:
        """
        Count images in folder without loading them

        Args:
            folder_path: Path to folder
            recursive: Search recursively

        Returns:
            Number of images found
        """
        try:
            images = ImageUtils.find_images_in_folder(folder_path, recursive=recursive)
            return len(images)
        except (NotFoundError, ValidationError):
            return 0


__all__ = ['ImageUtils']
