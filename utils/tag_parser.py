"""
Tag Parser - Extract classification tags from Claude's critique analysis output

This parser extracts the three classification tags (family, dynamic, reward) from
Claude's verbose critique analysis text by identifying positive indicators and
scoring each possible value.
"""
import re
import logging
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import yaml

from core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class TagParser:
    """Parser for extracting classification tags from Claude's critique analysis"""

    def __init__(self, config_path: str = "config/event_about_template.yaml"):
        """
        Initialize tag parser with configuration

        Args:
            config_path: Path to YAML config file containing field definitions

        Raises:
            ValidationError: If config file cannot be loaded
        """
        self.config_path = config_path
        self.possible_values = self._load_possible_values()

        # Positive indicators for selected values
        self.positive_indicators = [
            r"phù hợp nhất",
            r"được chọn",
            r"đúng",
            r"chính xác",
            r"rõ ràng",
            r"thấy rõ",
            r"là.*này",
            r"sự kiện này.*là",
            r"✓",
            r"✅"
        ]

        # Negative indicators for rejected values
        self.negative_indicators = [
            r"không phù hợp",
            r"không có",
            r"không thấy",
            r"không đúng",
            r"sai",
            r"✗",
            r"❌"
        ]

        logger.info("✅ TagParser initialized with config from %s", config_path)

    def _load_possible_values(self) -> Dict[str, List[str]]:
        """
        Load possible values from YAML config

        Returns:
            Dictionary mapping field names to lists of possible values

        Raises:
            ValidationError: If config cannot be loaded or is invalid
        """
        try:
            config_file = Path(self.config_path)

            if not config_file.exists():
                raise ValidationError(
                    message=f"Config file not found: {self.config_path}",
                    details={
                        "config_path": self.config_path,
                        "resolved_path": str(config_file.absolute())
                    }
                )

            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            field_defs = config.get('field_definitions', {})

            possible_values = {
                'family': field_defs.get('mechanic_family', {}).get('values', []),
                'dynamic': field_defs.get('player_dynamics', {}).get('values', []),
                'reward': field_defs.get('reward_types', {}).get('values', [])
            }

            # Validate that all fields have values
            for field_name, values in possible_values.items():
                if not values:
                    raise ValidationError(
                        message=f"No values found for field: {field_name}",
                        details={
                            "field_name": field_name,
                            "config_path": self.config_path
                        }
                    )

            logger.info(
                "✅ Loaded possible values: %d family, %d dynamic, %d reward",
                len(possible_values['family']),
                len(possible_values['dynamic']),
                len(possible_values['reward'])
            )

            return possible_values

        except yaml.YAMLError as e:
            raise ValidationError(
                message=f"Invalid YAML in config file: {self.config_path}",
                details={
                    "config_path": self.config_path,
                    "yaml_error": str(e)
                }
            )
        except Exception as e:
            raise ValidationError(
                message=f"Failed to load config: {str(e)}",
                details={
                    "config_path": self.config_path,
                    "error_type": type(e).__name__
                }
            )

    def parse_tags(self, critique_text: str) -> Dict[str, Optional[str]]:
        """
        Parse classification tags from critique analysis text

        Args:
            critique_text: Full text output from Claude including critique section

        Returns:
            Dictionary with 'family', 'dynamic', 'reward' keys and selected values
            Returns None for any tag that couldn't be parsed

        Example:
            >>> parser.parse_tags(claude_output)
            {
                'family': 'Quests',
                'dynamic': 'Collaborative',
                'reward': 'Currencies & items'
            }
        """
        logger.info("🔍 Parsing tags from critique text...")

        try:
            # Extract critique section
            critique_section = self._extract_critique_section(critique_text)

            if not critique_section:
                logger.warning("⚠️ Could not find critique section in text")
                return {'family': None, 'dynamic': None, 'reward': None}

            # Parse each tag type
            family = self._parse_field(critique_section, 'family', 'Family')
            dynamic = self._parse_field(critique_section, 'dynamic', 'Dynamic')
            reward = self._parse_field(critique_section, 'reward', 'Reward')

            result = {
                'family': family,
                'dynamic': dynamic,
                'reward': reward
            }

            # Log results
            parsed_count = sum(1 for v in result.values() if v is not None)
            logger.info(
                "✅ Parsed %d/3 tags: family=%s, dynamic=%s, reward=%s",
                parsed_count,
                family or 'None',
                dynamic or 'None',
                reward or 'None'
            )

            return result

        except Exception as e:
            logger.error("❌ Error parsing tags: %s", e, exc_info=True)
            return {'family': None, 'dynamic': None, 'reward': None}

    def _extract_critique_section(self, text: str) -> Optional[str]:
        """
        Extract the critique analysis section from full text

        Args:
            text: Full text output from Claude

        Returns:
            Critique section text or None if not found
        """
        # Look for the critique section marker
        patterns = [
            r'\[PHÂN TÍCH PHẢN BIỆN\](.*)',
            r'\*\*PHÂN TÍCH PHẢN BIỆN\*\*(.*)',
            r'PHÂN TÍCH PHẢN BIỆN[:\s]+(.*)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _parse_field(self, critique_text: str, field_key: str, field_label: str) -> Optional[str]:
        """
        Parse a single field from critique text

        Args:
            critique_text: Critique section text
            field_key: Field key ('family', 'dynamic', 'reward')
            field_label: Field label in text ('Family', 'Dynamic', 'Reward')

        Returns:
            Selected value or None if parsing failed
        """
        try:
            # Extract field section
            field_section = self._extract_field_section(critique_text, field_label)

            if not field_section:
                logger.warning("⚠️ Could not find %s section", field_label)
                return None

            # Score each possible value
            possible_values = self.possible_values[field_key]
            scores = {}

            for value in possible_values:
                score = self._score_value(field_section, value)
                scores[value] = score
                logger.debug("   %s: %s = %.2f", field_key, value, score)

            # Get value with highest score
            if not scores:
                return None

            best_value = max(scores.items(), key=lambda x: x[1])

            # Only return if score is positive (indicating selection)
            if best_value[1] > 0:
                logger.info("✅ Selected %s: %s (score: %.2f)", field_key, best_value[0], best_value[1])
                return best_value[0]
            else:
                logger.warning("⚠️ No positive score for %s (best: %s with %.2f)", field_key, best_value[0], best_value[1])
                return None

        except Exception as e:
            logger.error("❌ Error parsing %s field: %s", field_key, e)
            return None

    def _extract_field_section(self, critique_text: str, field_label: str) -> Optional[str]:
        """
        Extract a specific field section from critique text

        Args:
            critique_text: Full critique text
            field_label: Field label to extract ('Family', 'Dynamic', 'Reward')

        Returns:
            Field section text or None
        """
        # Pattern to match field section
        pattern = rf'\*\*{field_label}[^:]*:\*\*(.*?)(?=\*\*[A-Z]|$)'

        match = re.search(pattern, critique_text, re.DOTALL | re.IGNORECASE)

        if match:
            return match.group(1).strip()

        return None

    def _score_value(self, field_section: str, value: str) -> float:
        """
        Score a value based on indicators in field section

        Args:
            field_section: Text section for this field
            value: Value to score

        Returns:
            Score (positive = selected, negative = rejected, 0 = unclear)
        """
        score = 0.0

        # Find mentions of this value
        value_pattern = rf'[-•]\s*{re.escape(value)}[:\s]+(.*?)(?=[-•]|$)'
        matches = re.finditer(value_pattern, field_section, re.DOTALL | re.IGNORECASE)

        for match in matches:
            analysis_text = match.group(1).lower()

            # Check positive indicators
            for indicator_pattern in self.positive_indicators:
                if re.search(indicator_pattern, analysis_text, re.IGNORECASE):
                    score += 2.0

            # Check negative indicators
            for indicator_pattern in self.negative_indicators:
                if re.search(indicator_pattern, analysis_text, re.IGNORECASE):
                    score -= 1.0

        return score


# Global instance
_tag_parser_instance: Optional[TagParser] = None


def get_tag_parser() -> TagParser:
    """
    Get or create global tag parser instance

    Returns:
        TagParser instance
    """
    global _tag_parser_instance

    if _tag_parser_instance is None:
        _tag_parser_instance = TagParser()

    return _tag_parser_instance
