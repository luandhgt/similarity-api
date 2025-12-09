"""
Determine Alternative Service

This service analyzes whether a new event is an alternative (similar/repeated version)
of existing candidate events using LLM analysis.
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml

from services.llm_provider_base import BaseLLMProvider
from services.database_service import DatabaseService

logger = logging.getLogger(__name__)


class DetermineAlternativeService:
    """Service for determining if a new event is an alternative of existing events"""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        database_service: DatabaseService
    ):
        """
        Initialize DetermineAlternativeService

        Args:
            llm_provider: LLM provider instance (Claude/ChatGPT based on config)
            database_service: Database service instance
        """
        self.llm_provider = llm_provider
        self.database_service = database_service

        # Load prompts from YAML config
        self.prompts = self._load_prompts()

        logger.info("✅ DetermineAlternativeService initialized")

    def _load_prompts(self) -> Dict[str, Any]:
        """Load determine alternative prompts from YAML configuration"""
        try:
            prompts_path = Path("config/determine_alternative_prompts.yaml")
            with open(prompts_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"✅ Loaded determine alternative prompts from {prompts_path}")
                return config.get('determine_alternative', {})
        except Exception as e:
            logger.error(f"❌ Failed to load determine alternative prompts: {e}")
            raise

    async def determine_alternative(
        self,
        game_code: str,
        new_event: Dict[str, str],
        candidate_events: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Determine if a new event is an alternative of candidate events

        Args:
            game_code: Game code identifier
            new_event: Dict with 'name' and 'about' keys
            candidate_events: List of dicts with 'code', 'name', 'about' keys

        Returns:
            Dict with success, new_event, and alternatives list
        """
        logger.info(f"🔍 Analyzing alternatives for: {new_event['name']}")
        logger.info(f"   Game: {game_code}")
        logger.info(f"   Candidates: {len(candidate_events)}")

        # Step 1: Call LLM for analysis
        logger.info("🔄 Sending to LLM for analysis...")
        llm_response = await self._analyze_with_llm(
            game_code=game_code,
            new_event=new_event,
            candidate_events=candidate_events
        )

        # Step 2: Parse LLM response
        logger.info("🔄 Parsing LLM response...")
        parsed_results = self._parse_llm_response(llm_response)

        # Step 3: Get images for candidate events
        logger.info("🔄 Fetching images for candidate events...")
        event_codes = [c['code'] for c in candidate_events]
        images_map = await self.database_service.get_images_for_events(event_codes)

        # Step 4: Build final response
        alternatives = []
        for i, candidate in enumerate(candidate_events):
            event_code = candidate['code']

            # Find matching result from LLM
            llm_result = self._find_result_by_index(parsed_results, i)

            # Get images for this event
            images = images_map.get(event_code, [])
            formatted_images = [
                {
                    "file_name": img.get("file_name", ""),
                    "file_path": img.get("file_path", "")
                }
                for img in images
            ]

            alternatives.append({
                "event_code": event_code,
                "event_name": candidate['name'],
                "is_alternative": llm_result.get('is_alternative', False),
                "score": llm_result.get('score', 0),
                "reason": llm_result.get('reason', 'Không thể phân tích'),
                "images": formatted_images
            })

        logger.info(f"✅ Analysis complete. Results:")
        for alt in alternatives:
            logger.info(f"   - {alt['event_name']}: score={alt['score']}, is_alternative={alt['is_alternative']}")

        return {
            "success": True,
            "new_event": {
                "name": new_event['name'],
                "about": new_event['about']
            },
            "alternatives": alternatives
        }

    async def _analyze_with_llm(
        self,
        game_code: str,
        new_event: Dict[str, str],
        candidate_events: List[Dict[str, str]]
    ) -> str:
        """
        Send events to LLM for alternative analysis

        Args:
            game_code: Game code
            new_event: New event dict
            candidate_events: List of candidate event dicts

        Returns:
            LLM response string
        """
        # Build prompts
        system_prompt = self.prompts.get('system', '')
        user_prompt_template = self.prompts.get('user', '')

        # Fill in placeholders
        user_prompt = user_prompt_template.format(
            game_code=game_code,
            new_event_name=new_event['name'],
            new_event_about=new_event['about'],
            candidate_1_name=candidate_events[0]['name'],
            candidate_1_about=candidate_events[0]['about'],
            candidate_2_name=candidate_events[1]['name'],
            candidate_2_about=candidate_events[1]['about']
        )

        # Call LLM API
        logger.info(f"🔄 Calling LLM API...")
        response = await self.llm_provider.generate_text(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=2000
        )

        logger.info(f"✅ LLM response received ({len(response)} characters)")
        return response

    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse LLM JSON response

        Args:
            response: Raw LLM response string

        Returns:
            List of result dicts with candidate_index, is_alternative, score, reason
        """
        try:
            # Extract JSON from response
            json_str = self._extract_json_from_response(response)

            # Parse JSON
            data = json.loads(json_str)

            # Handle both formats: {"results": [...]} or direct list
            if isinstance(data, dict) and 'results' in data:
                results = data['results']
            elif isinstance(data, list):
                results = data
            else:
                logger.error(f"❌ Unexpected response format: {type(data)}")
                return []

            # Validate and normalize results
            normalized = []
            for result in results:
                normalized.append({
                    "candidate_index": result.get('candidate_index', 0),
                    "is_alternative": result.get('is_alternative', False),
                    "score": int(result.get('score', 0)),
                    "reason": result.get('reason', '')
                })

            logger.info(f"✅ Parsed {len(normalized)} results from LLM response")
            return normalized

        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse LLM JSON response: {e}")
            logger.error(f"Response preview: {response[:500]}...")
            return []
        except Exception as e:
            logger.error(f"❌ Error parsing LLM response: {e}")
            return []

    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON from LLM response (handles markdown code blocks)

        Args:
            response: Raw LLM response

        Returns:
            Clean JSON string
        """
        response = response.strip()

        # Try to find JSON object in markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*\})\s*```', response)
        if json_match:
            logger.debug("✅ Extracted JSON from markdown code block")
            return json_match.group(1).strip()

        # Try to find raw JSON object
        if '{' in response:
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx + 1]
                logger.debug(f"✅ Extracted JSON object from position {start_idx} to {end_idx}")
                return json_str.strip()

        # Return original if no match
        logger.warning("⚠️ Could not extract JSON from response, returning as-is")
        return response

    def _find_result_by_index(
        self,
        results: List[Dict[str, Any]],
        index: int
    ) -> Dict[str, Any]:
        """
        Find result by candidate_index

        Args:
            results: List of parsed results
            index: Candidate index to find

        Returns:
            Result dict or default values
        """
        for result in results:
            if result.get('candidate_index') == index:
                return result

        # Return default if not found
        logger.warning(f"⚠️ No result found for candidate_index={index}")
        return {
            "candidate_index": index,
            "is_alternative": False,
            "score": 0,
            "reason": "Không có kết quả phân tích từ LLM"
        }


# ==================== Service Instance Management ====================

_service_instance: Optional[DetermineAlternativeService] = None


def get_determine_alternative_service() -> Optional[DetermineAlternativeService]:
    """Get singleton DetermineAlternativeService instance"""
    global _service_instance
    return _service_instance


def initialize_determine_alternative_service(
    llm_provider: BaseLLMProvider,
    database_service: DatabaseService
) -> DetermineAlternativeService:
    """
    Initialize the global DetermineAlternativeService instance

    Args:
        llm_provider: LLM provider instance (Claude/ChatGPT based on config)
        database_service: Database service instance

    Returns:
        DetermineAlternativeService instance
    """
    global _service_instance
    _service_instance = DetermineAlternativeService(
        llm_provider=llm_provider,
        database_service=database_service
    )
    logger.info("✅ Global DetermineAlternativeService instance initialized")
    return _service_instance
