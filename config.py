"""
Configuration Module

Centralized configuration management for the Image Similarity API.
Loads environment variables and provides typed configuration access.
"""

import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
import logging

# Determine which .env file to load based on ENV variable or default
env_mode = os.getenv('ENV', 'development')
env_file = f'.env.{env_mode}'

# Try to load environment-specific file first, fallback to .env
if Path(env_file).exists():
    load_dotenv(env_file)
    print(f"Loaded configuration from {env_file}")
elif Path('.env').exists():
    load_dotenv('.env')
    print("Loaded configuration from .env")
else:
    print("Warning: No .env file found, using environment variables")


class Config:
    """Main configuration class"""

    # =============================================================================
    # ENVIRONMENT MODE
    # =============================================================================
    ENV: str = os.getenv('ENV', 'development')
    IS_DEVELOPMENT: bool = ENV == 'development'
    IS_PRODUCTION: bool = ENV == 'production'

    # =============================================================================
    # API KEYS
    # =============================================================================
    VOYAGE_API_KEY: str = os.getenv('VOYAGE_API_KEY', '')
    CLAUDE_API_KEY: str = os.getenv('CLAUDE_API_KEY', '')

    # =============================================================================
    # DATABASE CONFIGURATION
    # =============================================================================
    DB_HOST: str = os.getenv('DB_HOST', 'localhost')
    DB_PORT: int = int(os.getenv('DB_PORT', '5432'))
    DB_USER: str = os.getenv('DB_USER', '')
    DB_PASS: str = os.getenv('DB_PASS', '')
    DB_NAME: str = os.getenv('DB_NAME', '')

    # =============================================================================
    # SERVER CONFIGURATION
    # =============================================================================
    API_HOST: str = os.getenv('API_HOST', '0.0.0.0')
    API_PORT: int = int(os.getenv('API_PORT', '8000'))
    API_RELOAD: bool = os.getenv('API_RELOAD', 'true').lower() == 'true'

    # =============================================================================
    # PATH CONFIGURATION
    # =============================================================================
    PROJECT_ROOT: Path = Path(__file__).parent.resolve()

    SHARED_UPLOADS_PATH: str = os.getenv(
        'SHARED_UPLOADS_PATH',
        str(PROJECT_ROOT / 'shared' / 'uploads')
    )
    MODELS_PATH: str = os.getenv(
        'MODELS_PATH',
        str(PROJECT_ROOT / 'models')
    )
    INDEX_PATH: str = os.getenv(
        'INDEX_PATH',
        str(PROJECT_ROOT / 'index')
    )
    LOGS_DIR: str = os.getenv(
        'LOGS_DIR',
        str(PROJECT_ROOT / 'logs')
    )

    # =============================================================================
    # PLACES365 MODEL CONFIGURATION
    # =============================================================================
    PLACES365_MODEL_ARCH: str = os.getenv('PLACES365_MODEL_ARCH', 'resnet50')
    PLACES365_MODEL_FILE: str = os.getenv(
        'PLACES365_MODEL_FILE',
        'resnet50_places365.pth.tar'
    )
    PLACES365_NUM_CLASSES: int = int(os.getenv('PLACES365_NUM_CLASSES', '365'))
    PLACES365_MODEL_PATH: Path = Path(MODELS_PATH) / PLACES365_MODEL_FILE

    # =============================================================================
    # FAISS INDEX CONFIGURATION
    # =============================================================================
    FAISS_INDEX_TYPE: str = os.getenv('FAISS_INDEX_TYPE', 'IndexFlatL2')
    FAISS_DIMENSION: int = int(os.getenv('FAISS_DIMENSION', '2048'))

    # =============================================================================
    # IMAGE PROCESSING CONFIGURATION
    # =============================================================================
    MAX_IMAGE_SIZE: int = int(os.getenv('MAX_IMAGE_SIZE', '10485760'))  # 10MB
    SUPPORTED_IMAGE_FORMATS: List[str] = os.getenv(
        'SUPPORTED_IMAGE_FORMATS',
        'jpg,jpeg,png,bmp,tiff,tif,webp'
    ).split(',')
    IMAGE_RESIZE_WIDTH: int = int(os.getenv('IMAGE_RESIZE_WIDTH', '224'))
    IMAGE_RESIZE_HEIGHT: int = int(os.getenv('IMAGE_RESIZE_HEIGHT', '224'))

    # =============================================================================
    # TEXT EMBEDDING CONFIGURATION
    # =============================================================================
    VOYAGE_MODEL: str = os.getenv('VOYAGE_MODEL', 'voyage-2')
    VOYAGE_INPUT_TYPE: str = os.getenv('VOYAGE_INPUT_TYPE', 'document')
    TEXT_EMBEDDING_DIMENSION: int = int(os.getenv('TEXT_EMBEDDING_DIMENSION', '1024'))

    # =============================================================================
    # SIMILARITY SEARCH CONFIGURATION
    # =============================================================================
    TOP_K_RESULTS: int = int(os.getenv('TOP_K_RESULTS', '10'))
    TEXT_SIMILARITY_THRESHOLD: float = float(os.getenv('TEXT_SIMILARITY_THRESHOLD', '0.7'))
    IMAGE_SIMILARITY_THRESHOLD: float = float(os.getenv('IMAGE_SIMILARITY_THRESHOLD', '0.8'))
    COMBINED_SIMILARITY_WEIGHT_TEXT: float = float(
        os.getenv('COMBINED_SIMILARITY_WEIGHT_TEXT', '0.5')
    )
    COMBINED_SIMILARITY_WEIGHT_IMAGE: float = float(
        os.getenv('COMBINED_SIMILARITY_WEIGHT_IMAGE', '0.5')
    )

    # =============================================================================
    # CLAUDE API CONFIGURATION
    # =============================================================================
    CLAUDE_MODEL: str = os.getenv('CLAUDE_MODEL', 'claude-sonnet-4-5-20250929')
    CLAUDE_MAX_TOKENS: int = int(os.getenv('CLAUDE_MAX_TOKENS', '8000'))  # Increased for long critique analysis
    CLAUDE_TEMPERATURE: float = float(os.getenv('CLAUDE_TEMPERATURE', '0.7'))
    CLAUDE_TIMEOUT: int = int(os.getenv('CLAUDE_TIMEOUT', '120'))

    # =============================================================================
    # LOGGING CONFIGURATION
    # =============================================================================
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_TO_FILE: bool = os.getenv('LOG_TO_FILE', 'true').lower() == 'true'
    LOG_FILE_PATH: str = os.getenv('LOG_FILE_PATH', 'logs/image-similarity-api.log')
    LOG_MAX_BYTES: int = int(os.getenv('LOG_MAX_BYTES', '10485760'))  # 10MB
    LOG_BACKUP_COUNT: int = int(os.getenv('LOG_BACKUP_COUNT', '5'))
    LOG_FORMAT: str = os.getenv(
        'LOG_FORMAT',
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # =============================================================================
    # PERFORMANCE CONFIGURATION
    # =============================================================================
    WORKER_TIMEOUT: int = int(os.getenv('WORKER_TIMEOUT', '300'))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv('MAX_CONCURRENT_REQUESTS', '10'))
    CACHE_ENABLED: bool = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
    CACHE_TTL: int = int(os.getenv('CACHE_TTL', '3600'))

    @classmethod
    def validate(cls) -> List[str]:
        """
        Validate required configuration values.

        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []

        # Check required API keys
        if not cls.VOYAGE_API_KEY:
            errors.append("VOYAGE_API_KEY is required")
        if not cls.CLAUDE_API_KEY:
            errors.append("CLAUDE_API_KEY is required")

        # Check required database credentials
        if not cls.DB_USER:
            errors.append("DB_USER is required")
        if not cls.DB_PASS:
            errors.append("DB_PASS is required")
        if not cls.DB_NAME:
            errors.append("DB_NAME is required")

        # Check paths exist or can be created
        for path_name, path_value in [
            ('MODELS_PATH', cls.MODELS_PATH),
            ('INDEX_PATH', cls.INDEX_PATH),
            ('LOGS_DIR', cls.LOGS_DIR),
        ]:
            path = Path(path_value)
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"{path_name} cannot be created: {e}")

        return errors

    @classmethod
    def get_db_config(cls) -> dict:
        """Get database configuration as dictionary"""
        return {
            'host': cls.DB_HOST,
            'port': cls.DB_PORT,
            'user': cls.DB_USER,
            'password': cls.DB_PASS,
            'database': cls.DB_NAME
        }

    @classmethod
    def setup_logging(cls) -> None:
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        log_path = Path(cls.LOG_FILE_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure logging
        log_level = getattr(logging, cls.LOG_LEVEL.upper(), logging.INFO)

        handlers = [logging.StreamHandler()]

        if cls.LOG_TO_FILE:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                cls.LOG_FILE_PATH,
                maxBytes=cls.LOG_MAX_BYTES,
                backupCount=cls.LOG_BACKUP_COUNT
            )
            handlers.append(file_handler)

        logging.basicConfig(
            level=log_level,
            format=cls.LOG_FORMAT,
            handlers=handlers
        )

    @classmethod
    def print_config(cls) -> None:
        """Print current configuration (for debugging)"""
        print("=" * 60)
        print("CURRENT CONFIGURATION")
        print("=" * 60)
        print(f"Environment: {cls.ENV}")
        print(f"API Host: {cls.API_HOST}:{cls.API_PORT}")
        print(f"Database: {cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}")
        print(f"Log Level: {cls.LOG_LEVEL}")
        print(f"Models Path: {cls.MODELS_PATH}")
        print(f"Index Path: {cls.INDEX_PATH}")
        print(f"Logs Dir: {cls.LOGS_DIR}")
        print("=" * 60)


# Create a singleton instance
config = Config()

# Validate configuration on import (optional - can be moved to app startup)
validation_errors = config.validate()
if validation_errors and config.ENV == 'production':
    raise ValueError(f"Configuration validation failed:\n" + "\n".join(validation_errors))
elif validation_errors:
    print(f"Configuration warnings:\n" + "\n".join(validation_errors))
