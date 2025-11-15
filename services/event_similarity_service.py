"""
Event Similarity Service - Simplified Text-Only Version

This service finds similar events using text-based FAISS search (name + about)
and Claude AI for taxonomy classification and similarity analysis.

Flow:
1. Search FAISS for top 10 from name index + top 10 from about index
2. Combine and deduplicate to get ~20 unique candidate events
3. Send all to Claude with taxonomy guides
4. Claude assigns taxonomy to all events and finds similar ones
5. Return formatted results with Vietnamese translation and reasoning
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml
import json
import re

from services.claude_service import ClaudeService
from services.database_service import DatabaseService
from utils.text_processor import extract_text_features, VoyageClient
from utils.faiss_manager import search_similar_vectors, normalize_game_code

logger = logging.getLogger(__name__)


class EventSimilarityService:
    """Service for finding similar events using text-based search and Claude AI"""

    def __init__(
        self,
        claude_service: ClaudeService,
        voyage_client: VoyageClient,
        database_service: DatabaseService
    ):
        """
        Initialize EventSimilarityService

        Args:
            claude_service: Claude AI service instance
            voyage_client: Voyage embedding client instance
            database_service: Database service instance
        """
        self.claude_service = claude_service
        self.voyage_client = voyage_client
        self.database_service = database_service

        # Load prompts from YAML config
        self.prompts = self._load_prompts()

        logger.info("✅ EventSimilarityService initialized (text-only mode)")

    def _load_prompts(self) -> Dict[str, Any]:
        """Load similarity prompts from YAML configuration"""
        try:
            prompts_path = Path("config/similarity_prompts.yaml")
            with open(prompts_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"✅ Loaded similarity prompts from {prompts_path}")
                return config.get('similarity', {})
        except Exception as e:
            logger.error(f"❌ Failed to load similarity prompts: {e}")
            raise

    async def find_similar_events(
        self,
        folder_name: str,
        game_code: str,
        event_name: str,
        about: str,
        image_count: int,
        shared_uploads_path: str = "/shared/uploads/"
    ) -> Dict[str, Any]:
        """
        Find similar events using text-based search and Claude analysis

        Args:
            folder_name: Name of folder containing event images (not used in text-only version)
            game_code: Game code identifier (e.g., 'candy_crush')
            event_name: Name of the query event
            about: Description of the query event
            image_count: Expected number of images (validation only, not used yet)
            shared_uploads_path: Base path for shared uploads (not used yet)

        Returns:
            Dict with query_event and similar_events

        Raises:
            ValueError: If parameters are invalid
            Exception: If search or analysis fails
        """
        logger.info(f"🔍 Finding similar events for: {event_name}")
        logger.info(f"📝 Game: {game_code}, About length: {len(about)} chars")

        # Normalize game code
        normalized_game_code = normalize_game_code(game_code)

        # Step 1: Extract text embeddings for query event
        logger.info("🔄 Extracting text embeddings for query event...")
        name_vector = extract_text_features(event_name, self.voyage_client)
        about_vector = extract_text_features(about, self.voyage_client)

        # Step 2: Search FAISS for similar events
        logger.info("🔄 Searching FAISS indices for similar events...")

        # Search name index (top 10)
        name_results = search_similar_vectors(
            vector=name_vector,
            content_type="name",
            game_code=normalized_game_code,
            top_k=10
        )
        logger.info(f"✅ Found {len(name_results)} results from name index")

        # Search about index (top 10)
        about_results = search_similar_vectors(
            vector=about_vector,
            content_type="about",
            game_code=normalized_game_code,
            top_k=10
        )
        logger.info(f"✅ Found {len(about_results)} results from about index")

        # Step 3: Combine and deduplicate results
        candidate_faiss_indices = self._combine_search_results(name_results, about_results)
        logger.info(f"✅ Combined to {len(candidate_faiss_indices)} unique candidates")

        if len(candidate_faiss_indices) == 0:
            logger.warning("⚠️ No similar events found in FAISS indices")
            return {
                "query_event": {
                    "name": event_name,
                    "about": about,
                    "tags": {},
                    "tag_explanation": "No similar events found to perform taxonomy analysis"
                },
                "similar_events": []
            }

        # Step 4: Get event details from database
        logger.info(f"🔄 Fetching event details from database for {len(candidate_faiss_indices)} candidates...")
        candidate_events = await self.database_service.get_events_by_faiss_indices(candidate_faiss_indices)
        logger.info(f"✅ Retrieved {len(candidate_events)} candidate events from database")

        # Step 5: Send to Claude for taxonomy analysis and similarity assessment
        logger.info("🔄 Sending to Claude for analysis...")
        claude_response = await self._analyze_with_claude(
            query_name=event_name,
            query_about=about,
            candidates=candidate_events
        )

        # Step 6: Parse and format Claude's response
        logger.info("🔄 Parsing Claude response...")
        result = self._parse_claude_response(claude_response, event_name, about)

        logger.info(f"✅ Analysis complete. Found {len(result['similar_events'])} similar events")
        return result

    def _combine_search_results(
        self,
        name_results: List[Dict],
        about_results: List[Dict]
    ) -> List[int]:
        """
        Combine and deduplicate FAISS search results from name and about indices

        Args:
            name_results: Results from name index search
            about_results: Results from about index search

        Returns:
            List of unique FAISS indices
        """
        # Collect unique FAISS indices
        indices_set = set()

        for result in name_results:
            indices_set.add(result['index'])

        for result in about_results:
            indices_set.add(result['index'])

        return list(indices_set)

    async def _analyze_with_claude(
        self,
        query_name: str,
        query_about: str,
        candidates: List[Dict[str, Any]]
    ) -> str:
        """
        Send query and candidates to Claude for taxonomy analysis and similarity assessment

        Args:
            query_name: Name of query event
            query_about: About text of query event
            candidates: List of candidate event dictionaries from database

        Returns:
            Claude's JSON response as string
        """
        # Format candidates for prompt
        candidates_text = self._format_candidates_for_prompt(candidates)

        # Build prompts
        system_prompt = self.prompts.get('system', '')
        user_prompt_template = self.prompts.get('user', '')

        # Fill in placeholders
        user_prompt = user_prompt_template.format(
            query_name=query_name,
            query_about=query_about,
            candidate_count=len(candidates),
            candidates=candidates_text
        )

        # Call Claude API
        logger.info(f"🔄 Calling Claude API with {len(candidates)} candidates...")
        response = await self.claude_service._make_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=8000
        )

        logger.info(f"✅ Claude response received ({len(response)} characters)")
        return response

    def _format_candidates_for_prompt(self, candidates: List[Dict[str, Any]]) -> str:
        """
        Format candidate events for Claude prompt

        Args:
            candidates: List of candidate event dictionaries

        Returns:
            Formatted string with candidate events
        """
        formatted_lines = []

        for i, candidate in enumerate(candidates, 1):
            # Extract event details
            event_name = candidate.get('event_name', 'Unknown')
            event_code = candidate.get('event_code', 'unknown')

            # Get about text from text_embeddings
            text_embeddings = candidate.get('text_embeddings', {})
            about_text = text_embeddings.get('about', {}).get('text_content', '[No about text]')
            name_text = text_embeddings.get('name', {}).get('text_content', event_name)

            formatted_lines.append(
                f"{i}. Event Code: {event_code}\n"
                f"   Name: {name_text}\n"
                f"   About: {about_text}\n"
            )

        return "\n".join(formatted_lines)

    def _parse_claude_response(
        self,
        claude_response: str,
        query_name: str,
        query_about: str
    ) -> Dict[str, Any]:
        """
        Parse Claude's JSON response into the required format

        Args:
            claude_response: Raw Claude response string
            query_name: Query event name
            query_about: Query event about text

        Returns:
            Formatted response dictionary
        """
        try:
            # Extract JSON from response (Claude might wrap it in markdown code blocks)
            json_str = self._extract_json_from_response(claude_response)

            # Parse JSON
            similar_events = json.loads(json_str)

            # Validate that it's a list
            if not isinstance(similar_events, list):
                logger.error("❌ Claude response is not a JSON array")
                similar_events = []

            # Build response in expected format
            result = {
                "query_event": {
                    "name": query_name,
                    "about": query_about,
                    "tags": self._extract_taxonomy_from_first_result(similar_events),
                    "tag_explanation": "Taxonomy assigned by Claude based on event analysis"
                },
                "similar_events": similar_events
            }

            return result

        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse Claude JSON response: {e}")
            logger.error(f"Response preview: {claude_response[:500]}...")
            return {
                "query_event": {
                    "name": query_name,
                    "about": query_about,
                    "tags": {},
                    "tag_explanation": "Failed to parse Claude response"
                },
                "similar_events": []
            }
        except Exception as e:
            logger.error(f"❌ Error parsing Claude response: {e}")
            return {
                "query_event": {
                    "name": query_name,
                    "about": query_about,
                    "tags": {},
                    "tag_explanation": f"Error: {str(e)}"
                },
                "similar_events": []
            }

    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON from Claude response (handles markdown code blocks)

        Args:
            response: Raw Claude response

        Returns:
            Clean JSON string
        """
        # Try to find JSON in markdown code block
        json_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', response)
        if json_match:
            return json_match.group(1)

        # Try to find raw JSON array
        json_match = re.search(r'(\[[\s\S]*\])', response)
        if json_match:
            return json_match.group(1)

        # Return original if no match
        return response.strip()

    def _extract_taxonomy_from_first_result(self, similar_events: List[Dict]) -> Dict:
        """
        Extract taxonomy tags from first similar event's detail field

        Args:
            similar_events: List of similar events

        Returns:
            Dictionary with family, dynamics, rewards keys
        """
        if not similar_events:
            return {}

        try:
            # Get first event's detail
            first_event = similar_events[0]
            detail = first_event.get('detail', '')

            # Extract Taxonomy section
            taxonomy_match = re.search(r'\*\*Taxonomy:\*\*\s*(.*?)(?:\*\*|$)', detail, re.DOTALL)
            if not taxonomy_match:
                return {}

            taxonomy_text = taxonomy_match.group(1)

            # Extract individual taxonomy values
            family_match = re.search(r'Family:\s*([^\n]+)', taxonomy_text)
            dynamics_match = re.search(r'Dynamics:\s*([^\n]+)', taxonomy_text)
            rewards_match = re.search(r'Rewards:\s*([^\n]+)', taxonomy_text)

            return {
                "family": family_match.group(1).strip() if family_match else "",
                "dynamics": dynamics_match.group(1).strip() if dynamics_match else "",
                "rewards": rewards_match.group(1).strip() if rewards_match else ""
            }

        except Exception as e:
            logger.error(f"❌ Error extracting taxonomy: {e}")
            return {}

    async def get_service_status(self) -> Dict[str, Any]:
        """
        Get comprehensive service status

        Returns:
            Status dictionary with database and AI models info
        """
        try:
            # Check database connection
            db_health = await self.database_service.health_check()
            db_connected = db_health.get('status') == 'healthy'

            # Check Claude service
            claude_stats = self.claude_service.get_usage_stats()
            claude_ready = claude_stats.get('status') == 'ready'

            # Check Voyage client
            voyage_ready = self.voyage_client is not None

            return {
                "status": "healthy" if (db_connected and claude_ready and voyage_ready) else "degraded",
                "database_connected": db_connected,
                "faiss_indexes_loaded": {},  # TODO: Add FAISS status check
                "models_loaded": {
                    "claude": claude_ready,
                    "voyage": voyage_ready
                },
                "message": "Event similarity service ready (text-only mode)"
            }

        except Exception as e:
            logger.error(f"❌ Error checking service status: {e}")
            return {
                "status": "error",
                "database_connected": False,
                "faiss_indexes_loaded": {},
                "models_loaded": {
                    "claude": False,
                    "voyage": False
                },
                "message": f"Service status check failed: {str(e)}"
            }


# Singleton instance
_service_instance: Optional[EventSimilarityService] = None


def get_event_similarity_service() -> Optional[EventSimilarityService]:
    """Get singleton EventSimilarityService instance"""
    global _service_instance
    return _service_instance


def initialize_event_similarity_service(
    claude_service: ClaudeService,
    voyage_client: VoyageClient,
    database_service: DatabaseService
) -> EventSimilarityService:
    """
    Initialize the global EventSimilarityService instance

    Args:
        claude_service: Claude AI service instance
        voyage_client: Voyage embedding client instance
        database_service: Database service instance

    Returns:
        EventSimilarityService instance
    """
    global _service_instance
    _service_instance = EventSimilarityService(
        claude_service=claude_service,
        voyage_client=voyage_client,
        database_service=database_service
    )
    logger.info("✅ Global EventSimilarityService instance initialized")
    return _service_instance
