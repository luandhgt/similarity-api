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
import numpy as np
import asyncio

from services.claude_service import ClaudeService
from services.database_service import DatabaseService
from utils.text_processor import extract_text_features, VoyageClient
from utils.faiss_manager import search_similar_vectors, normalize_game_code, get_vector_by_index
from utils.image_processor import extract_image_features

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
        Find similar events using PARALLEL text + image search, then merge results

        Flow:
        1. Run text search (FAISS + Claude) and image search in PARALLEL
        2. Wait for both to complete
        3. Merge results (max 20 text + max 10 image = max 30 unique events)
        4. Enrich with missing data from database
        5. Return combined results with both text_score and image_score

        Args:
            folder_name: Name of folder containing event images
            game_code: Game code identifier (e.g., 'candy_crush')
            event_name: Name of the query event
            about: Description of the query event
            image_count: Expected number of images
            shared_uploads_path: Base path for shared uploads

        Returns:
            Dict with query_event and similar_events (with both scores)

        Raises:
            ValueError: If parameters are invalid
            Exception: If search or analysis fails
        """
        logger.info(f"🔍 Finding similar events for: {event_name}")
        logger.info(f"📝 Game: {game_code}, Folder: {folder_name}, Images: {image_count}")

        # Normalize game code
        normalized_game_code = normalize_game_code(game_code)

        # Get uploaded image paths
        folder_path = Path(shared_uploads_path) / folder_name
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        uploaded_image_paths = [
            str(f) for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if len(uploaded_image_paths) != image_count:
            logger.warning(f"⚠️ Image count mismatch: expected {image_count}, found {len(uploaded_image_paths)}")

        logger.info(f"✅ Found {len(uploaded_image_paths)} images in folder")

        # Run text and image search IN PARALLEL
        logger.info("🚀 Starting PARALLEL text + image search...")
        text_result, image_scores = await asyncio.gather(
            self._search_by_text(event_name, about, normalized_game_code),
            self._search_by_image(uploaded_image_paths, game_code, normalized_game_code, top_k=10)
        )

        logger.info("✅ Both searches complete. Merging results...")

        # Merge results
        merged_result = await self._merge_text_and_image_results(
            text_result=text_result,
            image_scores=image_scores
        )

        logger.info(f"✅ Final result: {len(merged_result['similar_events'])} events with combined scores")
        return merged_result

    async def _search_by_text(
        self,
        event_name: str,
        about: str,
        normalized_game_code: str
    ) -> Dict[str, Any]:
        """
        Text-only search (original implementation)

        Returns:
            Dict with query_event and similar_events (max 20, from Claude)
        """
        logger.info("🔄 [TEXT] Extracting text embeddings...")
        name_vector = extract_text_features(event_name, self.voyage_client)
        about_vector = extract_text_features(about, self.voyage_client)

        logger.info("🔄 [TEXT] Searching FAISS indices...")

        # Search name index (top 10)
        name_results = search_similar_vectors(
            vector=name_vector,
            content_type="name",
            game_code=normalized_game_code,
            top_k=10
        )
        logger.info(f"✅ [TEXT] Found {len(name_results)} from name index")

        # Search about index (top 10)
        about_results = search_similar_vectors(
            vector=about_vector,
            content_type="about",
            game_code=normalized_game_code,
            top_k=10
        )
        logger.info(f"✅ [TEXT] Found {len(about_results)} from about index")

        # Combine and deduplicate
        candidate_faiss_indices = self._combine_search_results(name_results, about_results)
        logger.info(f"✅ [TEXT] Combined to {len(candidate_faiss_indices)} unique candidates")

        if len(candidate_faiss_indices) == 0:
            logger.warning("⚠️ [TEXT] No similar events found")
            return {
                "query_event": {
                    "name": event_name,
                    "about": about
                },
                "similar_events": []
            }

        # Get event details from database
        logger.info(f"🔄 [TEXT] Fetching {len(candidate_faiss_indices)} candidates from database...")
        candidate_events = await self.database_service.get_events_by_faiss_indices(candidate_faiss_indices)
        logger.info(f"✅ [TEXT] Retrieved {len(candidate_events)} candidate events")

        # Send to Claude
        logger.info("🔄 [TEXT] Sending to Claude for analysis...")
        claude_response = await self._analyze_with_claude(
            query_name=event_name,
            query_about=about,
            candidates=candidate_events
        )

        # Parse response
        logger.info("🔄 [TEXT] Parsing Claude response...")
        result = self._parse_claude_response(claude_response, event_name, about)

        logger.info(f"✅ [TEXT] Search complete: {len(result['similar_events'])} events from Claude")
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

            # BACKUP FILTER: Only keep events with score >= 50 (in case Claude didn't filter)
            original_count = len(similar_events)
            similar_events = [event for event in similar_events if event.get('score', 0) >= 50]
            filtered_count = original_count - len(similar_events)

            if filtered_count > 0:
                logger.info(f"🔄 [FILTER] Removed {filtered_count} low-score events (score < 50). Kept {len(similar_events)}/{original_count}")

            # Build response in expected format
            result = {
                "query_event": {
                    "name": query_name,
                    "about": query_about
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
                    "about": query_about
                },
                "similar_events": []
            }
        except Exception as e:
            logger.error(f"❌ Error parsing Claude response: {e}")
            return {
                "query_event": {
                    "name": query_name,
                    "about": query_about
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

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    async def _search_by_image(
        self,
        uploaded_image_paths: List[str],
        game_code: str,
        normalized_game_code: str,
        top_k: int = 10
    ) -> Dict[str, float]:
        """
        Search for similar events by comparing uploaded images with all event images

        Algorithm (brute force):
        - FOR each event in the game
          - FOR each uploaded image
            - Find best matching image in that event (max similarity)
          - Calculate average across all uploaded images
        - Return top K events by average score

        Args:
            uploaded_image_paths: List of paths to uploaded query images
            game_code: Original game code for database queries (e.g., "Candy Crush")
            normalized_game_code: Normalized game code for FAISS index files (e.g., "candy_crush")
            top_k: Number of top events to return

        Returns:
            Dict mapping event_code to average image similarity score (0-1)
        """
        logger.info(f"🔍 Starting image search with {len(uploaded_image_paths)} uploaded images")

        # Step 1: Extract vectors from uploaded images
        logger.info("🔄 Extracting features from uploaded images...")
        uploaded_vectors = []
        for img_path in uploaded_image_paths:
            try:
                vector = extract_image_features(img_path)
                uploaded_vectors.append(vector)
                logger.debug(f"✅ Extracted features from {Path(img_path).name}")
            except Exception as e:
                logger.error(f"❌ Failed to extract features from {img_path}: {e}")
                # Continue with other images
                continue

        if len(uploaded_vectors) == 0:
            logger.error("❌ No valid uploaded image vectors")
            return {}

        logger.info(f"✅ Extracted {len(uploaded_vectors)} uploaded image vectors")

        # Step 2: Get all events for this game
        logger.info(f"🔄 Fetching all events for game {game_code}...")
        all_events = await self.database_service.get_all_events_for_game(game_code)
        logger.info(f"✅ Found {len(all_events)} events for game {game_code}")

        if len(all_events) == 0:
            logger.warning(f"⚠️ No events found for game {game_code}")
            return {}

        # Step 3: Calculate similarity scores for each event
        logger.info("🔄 Calculating image similarities (brute force)...")
        event_scores = {}

        # DEBUG: Track "Legendary Tavern" events for comparison
        legendary_tavern_matches = []

        for i, event in enumerate(all_events, 1):
            event_code = event['event_code']
            event_name = event.get('event_name', '')  # Fix: use 'event_name' not 'name'

            # Get all image FAISS indices for this event
            image_indices = await self.database_service.get_image_faiss_indices_for_event(
                event_code=event_code
            )

            if len(image_indices) == 0:
                # Skip events without images
                continue

            # Get vectors from FAISS for this event's images
            event_vectors = []
            for faiss_idx in image_indices:
                try:
                    # Use normalized_game_code for FAISS index file operations
                    vector = get_vector_by_index(faiss_idx, "images", normalized_game_code)
                    if vector is not None:
                        event_vectors.append(vector)
                except Exception as e:
                    logger.error(f"❌ Error getting vector {faiss_idx}: {e}")
                    continue

            if len(event_vectors) == 0:
                continue

            # Calculate similarity: for each uploaded image, find best match in event
            uploaded_best_scores = []
            for uploaded_vec in uploaded_vectors:
                # Find max similarity with any image in this event
                max_score = 0.0
                for event_vec in event_vectors:
                    similarity = self._cosine_similarity(uploaded_vec, event_vec)
                    max_score = max(max_score, similarity)

                uploaded_best_scores.append(max_score)

            # Average across all uploaded images
            avg_score = sum(uploaded_best_scores) / len(uploaded_best_scores)
            event_scores[event_code] = avg_score

            # DEBUG: Check if this is "Legendary Tavern - Rise of Kingdoms"
            if "Legendary Tavern" in event_name:
                legendary_tavern_matches.append({
                    'event_code': event_code,
                    'event_name': event_name,
                    'score': avg_score
                })
                logger.info(f"🎯 [DEBUG] Found Legendary Tavern: {event_name} | Score: {avg_score:.4f} ({avg_score*100:.2f}%)")

            if i % 100 == 0:
                logger.info(f"   Processed {i}/{len(all_events)} events...")

        logger.info(f"✅ Calculated scores for {len(event_scores)} events with images")

        # Step 4: Sort and return top K
        sorted_events = sorted(event_scores.items(), key=lambda x: x[1], reverse=True)
        top_events = dict(sorted_events[:top_k])

        # DEBUG: Add "Legendary Tavern" matches to the end if not already in top K
        if legendary_tavern_matches:
            logger.info(f"🎯 [DEBUG] Found {len(legendary_tavern_matches)} Legendary Tavern event(s)")
            for match in legendary_tavern_matches:
                if match['event_code'] not in top_events:
                    # Add to the end
                    top_events[match['event_code']] = match['score']
                    logger.info(f"🎯 [DEBUG] Added Legendary Tavern to results (not in top {top_k}): {match['event_name']} | Score: {match['score']:.4f} ({match['score']*100:.2f}%)")
                else:
                    logger.info(f"🎯 [DEBUG] Legendary Tavern already in top {top_k}: {match['event_name']} | Score: {match['score']:.4f} ({match['score']*100:.2f}%)")
        else:
            logger.warning(f"⚠️ [DEBUG] No 'Legendary Tavern' event found in {game_code}")

        logger.info(f"✅ Image search complete. Returning {len(top_events)} events (top {top_k} + debug events)")
        return top_events

    async def _merge_text_and_image_results(
        self,
        text_result: Dict[str, Any],
        image_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Merge text and image search results

        Strategy:
        - Collect all unique event_codes from both text and image results
        - Fetch full details (name, about, images) from database for ALL events
        - For each event, determine if it's in text, image, or both
        - Build merged event with appropriate scores and reason
        - Sort by text_score first, then image_score (both descending)

        Args:
            text_result: Result from _search_by_text (Claude response)
            image_scores: Dict {event_code: image_score} from _search_by_image
            game_code: Original game code for database queries (e.g., "Candy Crush")
            normalized_game_code: Normalized game code for FAISS (e.g., "candy_crush")

        Returns:
            Merged result with both scores
        """
        logger.info("🔄 [MERGE] Merging text and image results...")

        # Extract text events (Claude returns event-code directly)
        text_events = text_result.get('similar_events', [])

        logger.info(f"   Text results: {len(text_events)} events")
        logger.info(f"   Image results: {len(image_scores)} events")

        # Collect all unique event_codes from BOTH text and image results
        all_event_codes = set(image_scores.keys())  # Image event codes

        # Build text events lookup map
        text_events_map = {}
        for text_event in text_events:
            event_code = text_event.get('event-code')
            if event_code:
                all_event_codes.add(event_code)
                text_events_map[event_code] = text_event
            else:
                logger.warning(f"⚠️ [MERGE] Text event missing event-code: {text_event.get('event-name', 'Unknown')}")

        # Get event details from database for ALL event codes (text + image)
        logger.info(f"🔄 [MERGE] Fetching details for {len(all_event_codes)} unique events...")
        events_details = await self.database_service.get_events_by_codes(list(all_event_codes))

        # Build events_details lookup map
        events_details_map = {}
        for event_detail in events_details:
            events_details_map[event_detail.get('code')] = event_detail

        # Get images for all events
        images_map = await self.database_service.get_images_for_events(list(all_event_codes))

        # Build merged events list
        merged_events = []

        # Process ALL event_codes
        for event_code in all_event_codes:
            # Check if event_code is in text or image results
            in_text = event_code in text_events_map
            in_image = event_code in image_scores

            # Get event details from database
            event_detail = events_details_map.get(event_code)
            if not event_detail:
                logger.warning(f"⚠️ [MERGE] Could not find details for event_code: {event_code}")
                continue

            # Get name and about from event_detail
            name = event_detail.get('name_text', event_detail.get('name', ''))
            about = event_detail.get('about_text', '')

            # Get images
            images = images_map.get(event_code, [])

            # Build merged event based on where it appears
            if in_text and in_image:
                # BOTH TEXT AND IMAGE
                text_event = text_events_map[event_code]
                merged_event = {
                    "name": name,
                    "about": about,
                    "score_text": text_event.get('score', 0),  # 0-100 from Claude
                    "score_image": int(image_scores.get(event_code, 0.0) * 100),  # 0-1 → 0-100
                    "reason": text_event.get('detail', ''),  # Full detail from Claude
                    "images": images
                }

            elif in_text:
                # TEXT ONLY
                text_event = text_events_map[event_code]
                merged_event = {
                    "name": name,
                    "about": about,
                    "score_text": text_event.get('score', 0),  # 0-100 from Claude
                    "score_image": 0,  # No image score
                    "reason": text_event.get('detail', ''),  # Full detail from Claude
                    "images": images
                }

            elif in_image:
                # IMAGE ONLY
                merged_event = {
                    "name": name,
                    "about": about,
                    "score_text": 0,  # No text score
                    "score_image": int(image_scores.get(event_code, 0.0) * 100),  # 0-1 → 0-100
                    "reason": "",  # No Claude analysis
                    "images": images
                }

            else:
                # Should not happen
                logger.warning(f"⚠️ [MERGE] Event {event_code} not in text or image results")
                continue

            merged_events.append(merged_event)

        # Sort by text_score first, then image_score (both descending)
        merged_events.sort(key=lambda x: (x['score_text'], x['score_image']), reverse=True)

        logger.info(f"✅ [MERGE] Merged {len(merged_events)} total events")

        return {
            "query_event": text_result.get('query_event', {}),
            "similar_events": merged_events
        }

    def _extract_about_from_detail(self, detail: str) -> str:
        """Extract English about text from Claude detail field"""
        try:
            match = re.search(r'\*\*About \(English\):\*\*\s*(.*?)(?:\*\*|$)', detail, re.DOTALL)
            if match:
                return match.group(1).strip()
            return ""
        except Exception:
            return ""

    def _extract_taxonomy_from_detail(self, detail: str) -> Dict[str, str]:
        """Extract taxonomy from Claude detail field"""
        try:
            taxonomy_match = re.search(r'\*\*Taxonomy:\*\*\s*(.*?)(?:\*\*|$)', detail, re.DOTALL)
            if not taxonomy_match:
                return {"family": "", "dynamics": "", "rewards": ""}

            taxonomy_text = taxonomy_match.group(1)

            family_match = re.search(r'Family:\s*([^\n]+)', taxonomy_text)
            dynamics_match = re.search(r'Dynamics:\s*([^\n]+)', taxonomy_text)
            rewards_match = re.search(r'Rewards:\s*([^\n]+)', taxonomy_text)

            return {
                "family": family_match.group(1).strip() if family_match else "",
                "dynamics": dynamics_match.group(1).strip() if dynamics_match else "",
                "rewards": rewards_match.group(1).strip() if rewards_match else ""
            }
        except Exception:
            return {"family": "", "dynamics": "", "rewards": ""}

    def _extract_tag_explanation_from_detail(self, detail: str) -> str:
        """Extract tag explanation from Claude detail field"""
        try:
            match = re.search(r'\*\*Lý giải:\*\*\s*(.*?)$', detail, re.DOTALL)
            if match:
                return match.group(1).strip()
            return ""
        except Exception:
            return ""

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
