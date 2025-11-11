"""
Test Configuration Module

Contains test configuration and validation logic.
"""

from pathlib import Path
from typing import Dict, Any


class TestConfig:
    """Test configuration and constants"""

    # Event Information
    EVENT_NAME = "Trojan Treasures "
    GAME_CODE = "Rise of Kingdoms"
    EVENT_ABOUT = """
# Trojan Treasures - About

**As war looms ever closer, unearth the hidden treasures of Ilios.**

## Overview
Open chests to earn tons of rewards using Trojan Keys in this exciting treasure hunt event.

## How to Participate
Use Trojan Keys to open chests and earn valuable rewards. You can draw individual chests (1 key) or perform bulk draws (10 keys) for greater efficiency.

## Getting Trojan Keys
- **Free Keys**: Limited stock of 10 free keys available
- **Gem Purchase**: 600 gems for additional keys (Stock: 200)
- **Bundle Offers**:
  - **Trojan Keys I Bundle**: $4.99 - Instantly receive 1,050 keys (1030% value, Stock: 1)
  - **Trojan Keys II Bundle**: $9.99 - Instantly receive 2,200 keys (577% value, Stock: 1)

## Rewards & Probabilities
The treasure chests contain the following rewards with their respective drop rates:

- **Hector Sculpture x10** - 1.398%
- **Hector Sculpture x1** - 17.482%
- **Dazzling Starlight Sculpture x4** - 12.587%
- **8-Hour Training Speedup x4** - 13.986%
- **3-Hour Training Speedup x4** - 20.979%
- **Level 4 Tome of Knowledge x4** - 20.979%
- **Level 4 "Pick One" Resource Chest x4** - 12.587%

## Important Notes
1. You can purchase Trojan Keys from the event shop using gems or real money bundles
2. Unused Trojan Keys will be converted into resource items at the end of the event - make sure to use them before the event expires!

Don't miss this opportunity to collect powerful Hector sculptures and valuable resources to strengthen your kingdom!

"""

    # Image Folder Configuration
    FOLDER_NAME = "trojan_treasure_rise_of_kingdoms"
    SHARED_UPLOADS_PATH = "/media/luanpc/Video/shared/uploads/"
    EXPECTED_IMAGE_COUNT = 4

    # Test Options
    VERBOSE_OUTPUT = True
    SAVE_RESULTS = True
    OUTPUT_FILE = "test_results.json"


def validate_test_configuration(config: TestConfig) -> int:
    """
    Validate test configuration and folder existence

    Args:
        config: TestConfig instance

    Returns:
        Actual image count found (0 if validation failed)
    """
    print("🔍 Validating test configuration...")

    import os
    from pathlib import Path

    # Check required environment variables
    required_env_vars = ['CLAUDE_API_KEY', 'VOYAGE_API_KEY', 'DB_HOST', 'DB_USER', 'DB_PASS', 'DB_NAME']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        return 0

    # Check image folder
    folder_path = Path(config.SHARED_UPLOADS_PATH) / config.FOLDER_NAME
    if not folder_path.exists():
        print(f"❌ Image folder not found: {folder_path}")
        print(f"   Please create the folder and add {config.EXPECTED_IMAGE_COUNT} test images")
        return 0

    # Check images in folder
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if len(image_files) == 0:
        print(f"❌ No image files found in folder: {folder_path}")
        return 0

    if len(image_files) != config.EXPECTED_IMAGE_COUNT:
        print(f"⚠️ Image count mismatch: expected {config.EXPECTED_IMAGE_COUNT}, found {len(image_files)}")
        print(f"   Found images: {[f.name for f in image_files]}")
        print(f"   Continuing with actual count: {len(image_files)}")

    print(f"✅ Configuration valid")
    print(f"   Event: {config.EVENT_NAME}")
    print(f"   Game: {config.GAME_CODE}")
    print(f"   Images: {len(image_files)} files in {folder_path}")

    return len(image_files)


def get_test_request(config: TestConfig, actual_image_count: int) -> Dict[str, Any]:
    """
    Prepare test request parameters

    Args:
        config: TestConfig instance
        actual_image_count: Actual number of images found

    Returns:
        Dictionary with test request parameters
    """
    return {
        "folder_name": config.FOLDER_NAME,
        "game_code": config.GAME_CODE,
        "event_name": config.EVENT_NAME,
        "about": config.EVENT_ABOUT,
        "image_count": actual_image_count,
        "shared_uploads_path": config.SHARED_UPLOADS_PATH
    }
