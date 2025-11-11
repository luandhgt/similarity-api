"""
Unit Tests for DTOs

Tests for Data Transfer Objects.
"""

import pytest
from models.dtos import (
    EventTagsDTO,
    EventDTO,
    SimilarEventDTO,
    SearchResultDTO,
    SimilaritySearchRequestDTO
)


class TestEventTagsDTO:
    """Test EventTagsDTO"""

    def test_create_tags(self):
        """Test creating event tags"""
        tags = EventTagsDTO(
            family="Treasure Hunt",
            dynamics="Gacha",
            rewards="Sculptures"
        )

        assert tags.family == "Treasure Hunt"
        assert tags.dynamics == "Gacha"
        assert tags.rewards == "Sculptures"

    def test_to_dict(self):
        """Test converting tags to dict"""
        tags = EventTagsDTO(
            family="Treasure Hunt",
            dynamics="Gacha",
            rewards="Sculptures"
        )

        result = tags.to_dict()

        assert result['family'] == "Treasure Hunt"
        assert result['dynamics'] == "Gacha"
        assert result['rewards'] == "Sculptures"

    def test_from_dict(self):
        """Test creating tags from dict"""
        data = {
            'family': 'Treasure Hunt',
            'dynamics': 'Gacha',
            'rewards': 'Sculptures'
        }

        tags = EventTagsDTO.from_dict(data)

        assert tags.family == "Treasure Hunt"
        assert tags.dynamics == "Gacha"


class TestEventDTO:
    """Test EventDTO"""

    def test_create_event(self):
        """Test creating event"""
        event = EventDTO(
            name="Trojan Treasures",
            about="A treasure hunt event",
            game_code="Rise of Kingdoms"
        )

        assert event.name == "Trojan Treasures"
        assert event.about == "A treasure hunt event"
        assert event.game_code == "Rise of Kingdoms"

    def test_event_with_tags(self):
        """Test event with tags"""
        tags = EventTagsDTO(
            family="Treasure Hunt",
            dynamics="Gacha",
            rewards="Sculptures"
        )

        event = EventDTO(
            name="Trojan Treasures",
            about="A treasure hunt event",
            game_code="Rise of Kingdoms",
            tags=tags
        )

        assert event.tags is not None
        assert event.tags.family == "Treasure Hunt"

    def test_to_dict(self):
        """Test converting event to dict"""
        event = EventDTO(
            name="Trojan Treasures",
            about="A treasure hunt event",
            game_code="Rise of Kingdoms",
            faiss_index_name=42
        )

        result = event.to_dict()

        assert result['name'] == "Trojan Treasures"
        assert result['about'] == "A treasure hunt event"
        assert result['game_code'] == "Rise of Kingdoms"
        assert result['faiss_index_name'] == 42

    def test_from_dict(self):
        """Test creating event from dict"""
        data = {
            'name': 'Trojan Treasures',
            'about': 'A treasure hunt event',
            'game_code': 'Rise of Kingdoms',
            'faiss_index_name': 42
        }

        event = EventDTO.from_dict(data)

        assert event.name == "Trojan Treasures"
        assert event.faiss_index_name == 42


class TestSimilarEventDTO:
    """Test SimilarEventDTO"""

    def test_create_similar_event(self):
        """Test creating similar event"""
        event = EventDTO(
            name="Test Event",
            about="Test about",
            game_code="Test Game"
        )

        similar = SimilarEventDTO(
            event=event,
            text_score=0.85,
            image_score=0.92,
            reason="High similarity in rewards",
            matching_images_count=3
        )

        assert similar.event.name == "Test Event"
        assert similar.text_score == 0.85
        assert similar.image_score == 0.92
        assert similar.matching_images_count == 3

    def test_to_dict(self):
        """Test converting similar event to dict"""
        event = EventDTO(
            name="Test Event",
            about="Test about",
            game_code="Test Game"
        )

        similar = SimilarEventDTO(
            event=event,
            text_score=0.85,
            image_score=0.92,
            reason="High similarity"
        )

        result = similar.to_dict()

        assert result['name'] == "Test Event"
        assert result['score_text'] == 0.85
        assert result['score_image'] == 0.92
        assert result['reason'] == "High similarity"


class TestSimilaritySearchRequestDTO:
    """Test SimilaritySearchRequestDTO"""

    def test_create_request(self):
        """Test creating search request"""
        request = SimilaritySearchRequestDTO(
            query_name="Test Event",
            query_about="Test about",
            folder_name="test_folder",
            game_code="Test Game",
            shared_uploads_path="/path/to/uploads",
            image_count=5,
            top_k=10
        )

        assert request.query_name == "Test Event"
        assert request.image_count == 5
        assert request.top_k == 10

    def test_validate_valid_request(self):
        """Test validating valid request"""
        request = SimilaritySearchRequestDTO(
            query_name="Test Event",
            query_about="Test about",
            folder_name="test_folder",
            game_code="Test Game",
            shared_uploads_path="/path/to/uploads",
            image_count=5
        )

        errors = request.validate()
        assert len(errors) == 0

    def test_validate_empty_query_name(self):
        """Test validation fails for empty query_name"""
        request = SimilaritySearchRequestDTO(
            query_name="",
            query_about="Test about",
            folder_name="test_folder",
            game_code="Test Game",
            shared_uploads_path="/path/to/uploads",
            image_count=5
        )

        errors = request.validate()
        assert len(errors) > 0
        assert any("query_name" in error for error in errors)

    def test_validate_invalid_image_count(self):
        """Test validation fails for invalid image_count"""
        request = SimilaritySearchRequestDTO(
            query_name="Test Event",
            query_about="Test about",
            folder_name="test_folder",
            game_code="Test Game",
            shared_uploads_path="/path/to/uploads",
            image_count=0  # Invalid
        )

        errors = request.validate()
        assert len(errors) > 0
        assert any("image_count" in error for error in errors)


class TestSearchResultDTO:
    """Test SearchResultDTO"""

    def test_create_search_result(self):
        """Test creating search result"""
        result = SearchResultDTO(faiss_index=42, distance=0.15)

        assert result.faiss_index == 42
        assert result.distance == 0.15

    def test_from_tuple(self):
        """Test creating from tuple"""
        result = SearchResultDTO.from_tuple(42, 0.15)

        assert result.faiss_index == 42
        assert result.distance == 0.15

    def test_to_dict(self):
        """Test converting to dict"""
        result = SearchResultDTO(faiss_index=42, distance=0.15)
        data = result.to_dict()

        assert data['faiss_index'] == 42
        assert data['distance'] == 0.15
