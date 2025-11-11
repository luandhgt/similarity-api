"""
Pytest Configuration and Fixtures

Shared fixtures for all tests.
"""

import pytest
import asyncio
from pathlib import Path
from core.container import ServiceContainer


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def reset_container():
    """Reset service container before each test"""
    ServiceContainer.reset()
    yield
    ServiceContainer.reset()


@pytest.fixture
def sample_event_data():
    """Sample event data for testing"""
    return {
        'name': 'Trojan Treasures',
        'about': 'A treasure hunt event with rewards',
        'game_code': 'Rise of Kingdoms',
        'tags': {
            'family': 'Treasure Hunt',
            'dynamics': 'Gacha',
            'rewards': 'Sculptures'
        },
        'faiss_index_name': 42,
        'faiss_index_about': 43
    }


@pytest.fixture
def mock_database_service():
    """Mock database service"""
    class MockDatabaseService:
        def __init__(self):
            self.pool = None

        async def initialize(self):
            return True

        async def health_check(self):
            return {"status": "healthy"}

        async def close(self):
            pass

    return MockDatabaseService()


@pytest.fixture
def mock_claude_service():
    """Mock Claude service"""
    class MockClaudeService:
        def get_usage_stats(self):
            return {"model": "claude-test", "status": "ok"}

        async def analyze_similarity(self, *args, **kwargs):
            return {"score": 0.85, "reason": "Test similarity"}

    return MockClaudeService()


@pytest.fixture
def temp_image_folder(tmp_path):
    """Create temporary folder with mock images"""
    folder = tmp_path / "test_images"
    folder.mkdir()

    # Create mock image files
    for i in range(3):
        image_file = folder / f"image_{i}.jpg"
        image_file.write_text(f"mock image {i}")

    return folder


@pytest.fixture
def sample_validation_errors():
    """Sample validation errors"""
    return [
        "query_name cannot be empty",
        "image_count must be positive"
    ]
