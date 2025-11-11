"""
Unit Tests for Validators

Tests for request validation utilities.
"""

import pytest
from pathlib import Path
from core.exceptions import ValidationError, NotFoundError
from utils.validators import RequestValidator


class TestRequestValidator:
    """Test RequestValidator class"""

    def test_validate_game_code_valid(self):
        """Test valid game code"""
        result = RequestValidator.validate_game_code("Rise of Kingdoms")
        assert result == "Rise of Kingdoms"

    def test_validate_game_code_with_whitespace(self):
        """Test game code with whitespace"""
        result = RequestValidator.validate_game_code("  Rise of Kingdoms  ")
        assert result == "Rise of Kingdoms"

    def test_validate_game_code_empty(self):
        """Test empty game code raises error"""
        with pytest.raises(ValidationError) as exc_info:
            RequestValidator.validate_game_code("")
        assert "cannot be empty" in str(exc_info.value)

    def test_validate_game_code_whitespace_only(self):
        """Test whitespace-only game code raises error"""
        with pytest.raises(ValidationError):
            RequestValidator.validate_game_code("   ")

    def test_validate_event_name_valid(self):
        """Test valid event name"""
        result = RequestValidator.validate_event_name("Trojan Treasures")
        assert result == "Trojan Treasures"

    def test_validate_event_name_too_short(self):
        """Test event name too short"""
        with pytest.raises(ValidationError) as exc_info:
            RequestValidator.validate_event_name("", min_length=1)
        assert "cannot be empty" in str(exc_info.value)

    def test_validate_event_name_too_long(self):
        """Test event name too long"""
        long_name = "A" * 250
        with pytest.raises(ValidationError) as exc_info:
            RequestValidator.validate_event_name(long_name, max_length=200)
        assert "must not exceed 200" in str(exc_info.value)

    def test_validate_positive_integer_valid(self):
        """Test valid positive integer"""
        result = RequestValidator.validate_positive_integer(10, "count")
        assert result == 10

    def test_validate_positive_integer_too_small(self):
        """Test integer too small"""
        with pytest.raises(ValidationError) as exc_info:
            RequestValidator.validate_positive_integer(0, "count", min_value=1)
        assert "must be at least 1" in str(exc_info.value)

    def test_validate_positive_integer_too_large(self):
        """Test integer too large"""
        with pytest.raises(ValidationError) as exc_info:
            RequestValidator.validate_positive_integer(150, "count", min_value=1, max_value=100)
        assert "must not exceed 100" in str(exc_info.value)

    def test_validate_text_content_valid(self):
        """Test valid text content"""
        result = RequestValidator.validate_text_content("Some text", "about")
        assert result == "Some text"

    def test_validate_text_content_empty(self):
        """Test empty text raises error"""
        with pytest.raises(ValidationError):
            RequestValidator.validate_text_content("", "about")

    def test_validate_text_content_whitespace_stripped(self):
        """Test whitespace is stripped"""
        result = RequestValidator.validate_text_content("  Some text  ", "about")
        assert result == "Some text"

    def test_validate_output_format_valid(self):
        """Test valid output format"""
        available = ["json", "yaml", "default"]
        result = RequestValidator.validate_output_format("json", available)
        assert result == "json"

    def test_validate_output_format_invalid_uses_default(self):
        """Test invalid format returns default"""
        available = ["json", "yaml", "default"]
        result = RequestValidator.validate_output_format("xml", available, default="default")
        assert result == "default"


@pytest.mark.asyncio
class TestRequestValidatorAsync:
    """Test async validator methods"""

    async def test_placeholder(self):
        """Placeholder for async tests"""
        assert True
