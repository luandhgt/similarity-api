"""
Event Similarity Service - Part 1
Core methods and Image Score Mapping logic
"""

from collections import defaultdict
import logging
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import json

from services.claude_service import ClaudeService
from services.database_service import DatabaseService
from utils.faiss_manager import (
    search_similar_vectors,
    get_faiss_stats,
    get_vector_by_index,
    normalize_game_code
)
from utils.text_processor import VoyageClient
from utils.image_processor import extract_image_features
from utils.prompt_manager import PromptManager

from utils.text_processor import extract_text_features

logger = logging.getLogger(__name__)


class EventSimilarityService:
    """Service for finding similar events using multi-modal search and Claude analysis"""
    
    def __init__(self, claude_service: ClaudeService, voyage_client: VoyageClient, 
                 prompt_manager: PromptManager, database_service: DatabaseService):
        self.claude_service = claude_service
        self.voyage_client = voyage_client
        self.prompt_manager = prompt_manager
        self.database_service = database_service
        self.similarity_config = {
            "top_k": 10,
            "individual_search_k": 20
        }
    
    async def find_similar_events(
        self,
        query_name: str,
        query_about: str,
        folder_name: str,
        game_code: str,
        shared_uploads_path: str,
        image_count: int,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """MODIFIED: Main function using phase separation approach"""
        if top_k is None:
            top_k = self.similarity_config["top_k"]

        try:
            # Input validation
            if not query_name or not query_name.strip():
                raise ValueError("query_name cannot be empty")
            if not query_about or not query_about.strip():
                raise ValueError("query_about cannot be empty")
            if not folder_name or not folder_name.strip():
                raise ValueError("folder_name cannot be empty")
            if not game_code or not game_code.strip():
                raise ValueError("game_code cannot be empty")
            if not shared_uploads_path or not shared_uploads_path.strip():
                raise ValueError("shared_uploads_path cannot be empty")
            if image_count <= 0:
                raise ValueError("image_count must be positive")
            if top_k <= 0:
                raise ValueError("top_k must be positive")

            # Validate paths exist
            if not os.path.exists(shared_uploads_path):
                raise ValueError(f"shared_uploads_path does not exist: {shared_uploads_path}")
            folder_path = os.path.join(shared_uploads_path, folder_name)
            if not os.path.exists(folder_path):
                raise ValueError(f"Folder does not exist: {folder_path}")
            
            logger.info(f"Starting phase-separated similarity search for: {query_name[:50]}... in game: {game_code}")
            
            # PHASE 1: Text Processing
            logger.info("Phase 1: Processing text events with Claude")
            name_results = await self.search_by_name(query_name, game_code, top_k)
            about_results = await self.search_by_about(query_about, game_code, top_k)
            text_faiss_indices = self._combine_text_results_only(name_results, about_results)
            text_events = await self._process_text_events_with_claude(
                query_name, query_about, game_code, text_faiss_indices
            )
            # text_events = []
            
            # PHASE 2: Image Processing  
            logger.info("Phase 2: Processing image events separately")
            image_results_max, image_results_avg = await self.search_by_images_multi_score(
                shared_uploads_path, folder_name, image_count, game_code, top_k
            )
            image_events = await self._process_image_events_separately(image_results_max, image_results_avg)
            
            # PHASE 3: Final Merge
            logger.info("Phase 3: Merging text and image events")
            merged_events = self._merge_text_and_image_events(text_events, image_events)
            
            # Build final response
            response = self._build_multi_score_response(query_name, query_about, merged_events)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in find_similar_events: {e}")
            raise


    def validate_faiss_index(self, game_code: str, content_type: str) -> None:
        """
        Validate FAISS index exists for game_code and content_type
        Raises ValueError if index not found or empty
        """
        try:
            normalized_game_code = normalize_game_code(game_code)
            faiss_stats = get_faiss_stats(game_name=normalized_game_code, content_type=content_type)
            
            if not faiss_stats:
                raise ValueError(f"FAISS index not found for game '{game_code}' content_type '{content_type}'")
                
            total_vectors = faiss_stats.get("total_vectors", 0)
            if total_vectors == 0:
                raise ValueError(f"FAISS index empty for game '{game_code}' content_type '{content_type}' (0 vectors)")
                
            logger.debug(f"FAISS validation passed: {content_type} index for {game_code} has {total_vectors} vectors")
            
        except Exception as e:
            if "FAISS index" in str(e):
                raise  # Re-raise our custom errors
            else:
                raise ValueError(f"Cannot validate FAISS index for game '{game_code}' content_type '{content_type}': {e}")
    
    async def _load_images_from_folder(
        self, 
        shared_uploads_path: str, 
        folder_name: str, 
        expected_count: int
    ) -> List[np.ndarray]:
        """Load and validate images from shared uploads folder, return embeddings"""
        import glob
        
        folder_path = os.path.join(shared_uploads_path, folder_name)
        
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder not found: {folder_path}")
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        # Validate image count
        actual_count = len(image_files)
        if actual_count != expected_count:
            raise ValueError(f"Image count mismatch: expected {expected_count}, found {actual_count}")
        
        # Convert images to embeddings using Places365
        image_embeddings = []
        for image_path in image_files:
            try:
                # Pass image path directly to extract_image_features
                embedding = extract_image_features(image_path)
                if embedding is not None:
                    image_embeddings.append(embedding)
                else:
                    logger.warning(f"Failed to extract features from {image_path}")
            except Exception as e:
                logger.warning(f"Failed to load image {image_path}: {e}")
        
        if len(image_embeddings) != expected_count:
            logger.warning(f"Expected {expected_count} embeddings, got {len(image_embeddings)}")
        
        logger.info(f"Loaded {len(image_embeddings)} image embeddings from {folder_path}")
        return image_embeddings
    
    async def search_by_name(self, query_name: str, game_code: str, top_k: int) -> List[Dict[str, Any]]:
        """Search for similar events by name with validation"""
        try:
            # Step 1: Validate FAISS index exists
            self.validate_faiss_index(game_code, "name")

            full_name = f"{query_name} - {game_code}"
            
            # Step 2: Convert raw text to embedding
            name_embedding = extract_text_features(full_name, self.voyage_client)
            
            # Step 3: Search FAISS index
            results = search_similar_vectors(
                vector=name_embedding,
                content_type="name",
                game_code=normalize_game_code(game_code),
                top_k=top_k
            )
            
            logger.info(f"Found {len(results)} similar names for query: {query_name[:50]}...")
            return results
            
        except ValueError as ve:
            logger.error(f"Validation error in search_by_name: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error in search_by_name: {e}")
            raise ValueError(f"Name search failed for game '{game_code}': {e}")

    async def search_by_about(self, query_about: str, game_code: str, top_k: int) -> List[Dict[str, Any]]:
        """Search for similar events by about content with validation"""
        try:
            # Step 1: Validate FAISS index exists
            self.validate_faiss_index(game_code, "about")
                        
            # Step 2: Convert raw text to embedding
            about_embedding = extract_text_features(query_about, self.voyage_client)
            
            # Step 3: Search FAISS index
            results = search_similar_vectors(
                vector=about_embedding,
                content_type="about",
                game_code=normalize_game_code(game_code),
                top_k=top_k
            )
            
            logger.info(f"Found {len(results)} similar about texts for query: {query_about[:50]}...")
            return results
            
        except ValueError as ve:
            logger.error(f"Validation error in search_by_about: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error in search_by_about: {e}")
            raise ValueError(f"About search failed for game '{game_code}': {e}")

    async def _map_faiss_indices_to_event_codes(
        self, 
        faiss_indices: List[int], 
        game_code: str
    ) -> Dict[int, str]:
        """
        Map FAISS indices to event codes using database lookup
        REQUIRES: images table with faiss_index and event_code columns
        """
        try:
            if not faiss_indices:
                return {}
            
            # Use ANY array syntax - cleaner for asyncpg
            query = """
            SELECT i.faiss_index, i.event_code
            FROM images i
            INNER JOIN events e ON i.event_code = e.code
            WHERE i.faiss_index = ANY($1::int[])
            AND e.game_code = $2
            AND i.is_deleted = false
            """
            
            # Pass faiss_indices as array and game_code as second param
            params = [faiss_indices, game_code]
            results = await self.database_service.execute_query(query, params)
            
            # Build mapping dictionary
            mapping = {}
            for row in results:
                mapping[row['faiss_index']] = row['event_code']
            
            logger.info(f"Mapped {len(mapping)} FAISS indices to event codes")
            return mapping
            
        except Exception as e:
            logger.error(f"Error mapping FAISS indices to event codes: {e}")
            return {}

    async def _get_vector_from_faiss_index(self, faiss_index: int, content_type: str, game_code: str) -> Optional[np.ndarray]:
        """
        Retrieve a specific vector from FAISS index by its index number.
        """
        try:
            return get_vector_by_index(faiss_index, content_type, normalize_game_code(game_code))
            
        except ImportError:
            logger.error("get_vector_by_index method not implemented in faiss_manager.py")
            return None
        except Exception as e:
            logger.error(f"Error retrieving vector from FAISS index {faiss_index}: {e}")
            return None
    
    async def query_postgres_for_texts(self, faiss_indices: List[int]) -> List[Dict[str, Any]]:
        """
        Query PostgreSQL database for event texts by FAISS indices
        """
        if not faiss_indices:
            logger.warning("No FAISS indices provided for database query")
            return []
        
        try:
            logger.info(f"Querying database for {len(faiss_indices)} FAISS indices")
            
            # Get event data by FAISS indices
            results = await self.database_service.get_events_by_faiss_indices(faiss_indices)
            
            if not results:
                logger.warning(f"No database records found for FAISS indices: {faiss_indices[:10]}...")
                return []
            
            # Format results with new flat structure
            formatted_results = []
            for event_data in results:
                try:
                    # Validate required fields
                    event_code = event_data.get("event_code")
                    if not event_code:
                        logger.warning(f"Event missing event_code: {event_data}")
                        continue
                    
                    formatted_result = {
                        "event_code": event_code,
                        "event_name": event_data.get("event_name", "Unknown Event"),
                        "name_content": event_data.get("name", ""),
                        "about_content": event_data.get("about", ""),
                    }
                    
                    formatted_results.append(formatted_result)
                    
                except Exception as e:
                    logger.warning(f"Error formatting event data: {e}, skipping event")
                    continue
            
            logger.info(f"Successfully formatted {len(formatted_results)}/{len(results)} database records")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return []

    async def analyze_text_similarity_with_claude(
        self,
        query_name: str,
        query_about: str,
        game_code: str,
        candidate_texts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Use Claude to analyze text similarity with comprehensive taxonomy tagging and alternative assessment"""
        try:
            # Get similarity analysis prompts
            prompts = self.prompt_manager.get_similarity_prompts()
            
            # Prepare the comprehensive analysis prompt
            system_prompt = prompts.get("system", "")
            user_prompt = prompts.get("user", "").format(
                query_name=query_name,
                query_about=query_about,
                game_code=game_code,
                candidate_count=len(candidate_texts),
                candidates=self._format_candidates_for_claude(candidate_texts)
            )
            
            # Send to Claude for comprehensive taxonomy + alternative analysis
            response = await self.claude_service._make_request(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=8000  # Increased for comprehensive analysis
            )
            
            # Parse Claude's comprehensive response
            comprehensive_results = self._parse_claude_comprehensive_response(response, candidate_texts)
            
            logger.info(f"Claude performed comprehensive analysis on {len(candidate_texts)} candidates")
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Error in Claude comprehensive similarity analysis: {e}")
            return self._create_comprehensive_fallback(candidate_texts)
    
    def _format_candidates_for_claude(self, candidates: List[Dict[str, Any]]) -> str:
        """Format candidate texts for Claude analysis"""
        formatted = []
        for i, candidate in enumerate(candidates, 1):
            formatted.append(f"""
Candidate {i}:
Event Code: {candidate['event_code']}
Event Name: {candidate['event_name']}
Name Content: {candidate['name_content']}
About Content: {candidate['about_content']}
Author: {candidate.get('author', 'Unknown')}
Image Score: {candidate.get('score_image', 0.0):.3f}
Found By: {candidate.get('found_by', 'unknown')}
---
""")
        return "\n".join(formatted)
    
    def _parse_claude_comprehensive_response(
        self,
        claude_response: str,
        original_candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse Claude's comprehensive taxonomy and alternative analysis response"""
        try:
            # Clean and parse JSON response
            cleaned_response = claude_response.strip()
            
            # Handle markdown code blocks
            if "```json" in cleaned_response:
                json_start = cleaned_response.find("```json") + 7
                json_end = cleaned_response.find("```", json_start)
                if json_end > json_start:
                    cleaned_response = cleaned_response[json_start:json_end]
            
            # Find JSON object bounds
            json_start = cleaned_response.find('{')
            json_end = cleaned_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = cleaned_response[json_start:json_end]
                parsed_data = json.loads(json_str)
                
                # Extract query event taxonomy
                query_event_taxonomy = parsed_data.get("query_event_taxonomy", {})
                
                # Extract and format candidate analysis
                candidate_analysis = parsed_data.get("candidate_analysis", [])
                formatted_candidates = []
                
                for candidate in candidate_analysis:
                    if isinstance(candidate, dict):
                        # Extract key information from Claude's analysis
                        taxonomy = candidate.get("taxonomy", {})
                        alternative_assessment = candidate.get("alternative_assessment", {})
                        final_determination = alternative_assessment.get("final_determination", {})
                        
                        # Include ALL candidates - no filtering by threshold
                        similarity_score = final_determination.get("similarity_percentage", 0)
                        
                        formatted_candidate = {
                            "event_code": candidate.get("event_code"),
                            "event_name": candidate.get("event_name"),
                            "taxonomy": {
                                "family": taxonomy.get("family", {}).get("selected_value", "Other"),
                                "dynamics": taxonomy.get("dynamics", {}).get("selected_value", "Individualistic"),
                                "rewards": taxonomy.get("rewards", {}).get("selected_value", "None")
                            },
                            "tag_explanation": self._build_tag_explanation(taxonomy),
                            "score_text": similarity_score / 100.0,  # Convert to 0-1 scale
                            "reason": final_determination.get("detailed_reasoning", ""),
                            "confidence": final_determination.get("confidence_level", "low"),
                            "alternative_type": final_determination.get("primary_type", "none")
                        }
                        formatted_candidates.append(formatted_candidate)
                
                # Sort by similarity score descending
                formatted_candidates.sort(key=lambda x: x["score_text"], reverse=True)
                
                return {
                    "query_event_taxonomy": query_event_taxonomy,
                    "formatted_candidates": formatted_candidates
                }
            
            else:
                logger.warning("Could not find valid JSON in Claude response")
                return self._create_comprehensive_fallback(original_candidates)
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return self._create_comprehensive_fallback(original_candidates)
        except Exception as e:
            logger.error(f"Error parsing Claude comprehensive response: {e}")
            return self._create_comprehensive_fallback(original_candidates)
    
    def _build_tag_explanation(self, taxonomy: Dict[str, Any]) -> str:
        """Build tag explanation from taxonomy analysis"""
        explanations = []
        
        for category, data in taxonomy.items():
            if isinstance(data, dict) and "selected_value" in data and "reasoning" in data:
                value = data["selected_value"]
                reasoning = data["reasoning"]
                explanations.append(f"{category.capitalize()}: {value} - {reasoning}")
        
        return ". ".join(explanations)
    
    def _create_comprehensive_fallback(self, original_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create fallback results when Claude analysis fails"""
        fallback_candidates = []
        
        for candidate in original_candidates:
            fallback_candidate = {
                "event_code": candidate["event_code"],
                "event_name": candidate["event_name"],
                "taxonomy": {
                    "family": "Other",
                    "dynamics": "Individualistic", 
                    "rewards": "None"
                },
                "tag_explanation": "Analysis failed - using fallback classification",
                "score_text": 0.25,  # Low fallback score
                "reason": "Analysis unavailable due to processing error",
                "confidence": "low",
                "alternative_type": "none"
            }
            fallback_candidates.append(fallback_candidate)
        
        return {
            "query_event_taxonomy": {
                "family": {"selected_value": "Other", "reasoning": "Analysis failed"},
                "dynamics": {"selected_value": "Individualistic", "reasoning": "Analysis failed"},
                "rewards": {"selected_value": "None", "reasoning": "Analysis failed"}
            },
            "formatted_candidates": fallback_candidates
        }
    
    def _build_final_response(
        self,
        query_name: str,
        query_about: str,
        claude_analysis: Dict[str, Any],
        candidate_data_with_scores: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build final response in required format"""
        
        # Extract query event information
        query_taxonomy = claude_analysis.get("query_event_taxonomy", {}) if claude_analysis else {}
        query_tags = {
            "family": query_taxonomy.get("family", {}).get("selected_value", "Other"),
            "dynamics": query_taxonomy.get("dynamics", {}).get("selected_value", "Individualistic"),
            "rewards": query_taxonomy.get("rewards", {}).get("selected_value", "None")
        }
        
        # Build query tag explanation
        query_tag_explanation = ""
        if claude_analysis and query_taxonomy:
            explanations = []
            for category, data in query_taxonomy.items():
                if isinstance(data, dict) and "selected_value" in data and "reasoning" in data:
                    value = data["selected_value"]
                    reasoning = data["reasoning"]
                    explanations.append(f"{category.capitalize()}: {value} - {reasoning}")
            query_tag_explanation = ". ".join(explanations)
        # Build similar events - merge Claude analysis with pre-assigned image scores
        similar_events = []
        claude_candidates = claude_analysis.get("formatted_candidates", []) if claude_analysis else []

        # Create mapping for quick lookup
        candidate_lookup = {c['event_code']: c for c in candidate_data_with_scores}

        for claude_candidate in claude_candidates:
            event_code = claude_candidate.get("event_code")
            original_candidate = candidate_lookup.get(event_code, {})
            
            # Simple merge - image scores already assigned in _assign_simple_image_scores
            similar_event = {
                "name": claude_candidate["event_name"],
                "about": original_candidate.get("about_content", ""),
                "score_text": claude_candidate["score_text"],
                "score_image": original_candidate.get("score_image", 0.0),  # Already assigned correctly
                "reason": claude_candidate["reason"],
                "tags": claude_candidate["taxonomy"],
                "tag_explanation": claude_candidate["tag_explanation"],
                "image_faiss_indices": original_candidate.get("image_faiss_indices", [])
            }
            similar_events.append(similar_event)
        
        # Build final response
        response = {
            "query_event": {
                "name": query_name,
                "about": query_about,
                "tags": query_tags,
                "tag_explanation": query_tag_explanation or "Analysis not available"
            },
            "similar_events": similar_events
        }
        
        return response
    
    # COMPLETE THE find_similar_events METHOD FROM PART 1
    async def complete_find_similar_events(
        self,
        query_name: str,
        query_about: str,
        candidate_data_with_scores: List[Dict[str, Any]],
        game_code: str = "unknown" 
    ) -> Dict[str, Any]:
        """Complete the find_similar_events workflow with Claude analysis and response building"""
        try:
            # Step 6: Analyze with Claude for comprehensive taxonomy and alternative analysis
            claude_analysis = None
            if candidate_data_with_scores:
                claude_analysis = await self.analyze_text_similarity_with_claude(
                    query_name, query_about, game_code, candidate_data_with_scores
                )
            
            # Step 7: Build response structure
            response = self._build_final_response(
                query_name, query_about, claude_analysis, candidate_data_with_scores
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error completing find_similar_events: {e}")
            raise            
        except Exception as e:
            logger.error(f"Error completing find_similar_events: {e}")
            raise
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get service health status"""
        status = {
            "service": "event_similarity",
            "status": "operational",
            "database_health": {},
            "faiss_stats": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Get database health
            status["database_health"] = await self.database_service.health_check()
            
            # Get FAISS statistics
            for content_type in ["name", "about", "images"]:
                status["faiss_stats"][content_type] = get_faiss_stats(content_type)
        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            status["faiss_stats"] = {"error": str(e)}
        
        return status
    
    def update_similarity_config(self, config: Dict[str, Any]):
        """Update similarity search configuration"""
        self.similarity_config.update(config)
        logger.info(f"Updated similarity config: {self.similarity_config}")
    
    # UTILITY METHODS FOR DEBUGGING AND MONITORING
    
    def get_image_mapping_stats(self, candidate_data_with_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about image score mapping for debugging"""
        stats = {
            "total_candidates": len(candidate_data_with_scores),
            "found_by_image_search": 0,
            "found_by_text_search": 0,
            "avg_image_score_from_image_search": 0.0,
            "avg_image_score_from_text_search": 0.0,
            "candidates_with_image_indices": 0,
            "total_image_indices": 0
        }
        
        image_search_scores = []
        text_search_scores = []
        
        for candidate in candidate_data_with_scores:
            found_by = candidate.get("found_by", "unknown")
            image_score = candidate.get("score_image", 0.0)
            image_indices = candidate.get("image_faiss_indices", [])
            
            if found_by == "image_search":
                stats["found_by_image_search"] += 1
                image_search_scores.append(image_score)
            elif found_by == "text_search":
                stats["found_by_text_search"] += 1
                text_search_scores.append(image_score)
            
            if image_indices:
                stats["candidates_with_image_indices"] += 1
                stats["total_image_indices"] += len(image_indices)
        
        # Calculate averages
        if image_search_scores:
            stats["avg_image_score_from_image_search"] = sum(image_search_scores) / len(image_search_scores)
        
        if text_search_scores:
            stats["avg_image_score_from_text_search"] = sum(text_search_scores) / len(text_search_scores)
        
        return stats
    
    def validate_response_completeness(self, response: Dict[str, Any]) -> Dict[str, Any]:   
        """Validate that response has all required fields and proper scores"""
        validation = {
            "valid": True,
            "issues": [],
            "stats": {
                "total_similar_events": 0,
                "events_with_text_scores": 0,
                "events_with_image_scores": 0,
                "events_with_both_scores": 0,
                "avg_text_score": 0.0,
                "avg_image_score": 0.0
            }
        }
        
        # Check query_event structure
        if "query_event" not in response:
            validation["valid"] = False
            validation["issues"].append("Missing query_event")
        else:
            query_event = response["query_event"]
            required_fields = ["name", "about", "tags", "tag_explanation"]
            for field in required_fields:
                if field not in query_event:
                    validation["issues"].append(f"Missing query_event.{field}")
        
        # Check similar_events structure
        if "similar_events" not in response:
            validation["valid"] = False
            validation["issues"].append("Missing similar_events")
        else:
            similar_events = response["similar_events"]
            validation["stats"]["total_similar_events"] = len(similar_events)
            
            text_scores = []
            image_scores = []
            
            for i, event in enumerate(similar_events):
                required_fields = ["name", "about", "score_text", "score_image", "reason", "tags", "tag_explanation", "image_faiss_indices"]
                for field in required_fields:
                    if field not in event:
                        validation["issues"].append(f"Missing similar_events[{i}].{field}")
                
                # Check scores
                if "score_text" in event and event["score_text"] > 0:
                    validation["stats"]["events_with_text_scores"] += 1
                    text_scores.append(event["score_text"])
                
                if "score_image" in event and event["score_image"] > 0:
                    validation["stats"]["events_with_image_scores"] += 1
                    image_scores.append(event["score_image"])
                
                if ("score_text" in event and event["score_text"] > 0 and 
                    "score_image" in event and event["score_image"] > 0):
                    validation["stats"]["events_with_both_scores"] += 1
            
            # Calculate averages
            if text_scores:
                validation["stats"]["avg_text_score"] = sum(text_scores) / len(text_scores)
            if image_scores:
                validation["stats"]["avg_image_score"] = sum(image_scores) / len(image_scores)
        
        if validation["issues"]:
            validation["valid"] = False
        
        return validation

    async def search_by_images_multi_score(
        self, 
        shared_uploads_path: str, 
        folder_name: str, 
        image_count: int,
        game_code: str, 
        top_k: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        NEW: Single-phase image search with max and average pooling
        Returns: (max_pooling_results, avg_pooling_results)
        """
        try:
            # Step 1: Validate and load images
            self.validate_faiss_index(game_code, "images")
            query_images = await self._load_images_from_folder(
                shared_uploads_path, folder_name, image_count
            )
            
            if not query_images:
                raise ValueError(f"No valid images found in folder: {folder_name}")
            
            # Step 2: Individual FAISS searches for each query image
            all_individual_results = []
            individual_k = self.similarity_config.get("individual_search_k", 20)
            
            for query_idx, query_img in enumerate(query_images):
                results = search_similar_vectors(
                    vector=query_img,
                    content_type="images",
                    game_code=normalize_game_code(game_code),
                    top_k=individual_k
                )
                
                # Tag with query image index
                for result in results:
                    result['query_image_idx'] = query_idx
                
                all_individual_results.extend(results)
            
            # Step 3: Group by event and calculate pooling scores
            max_results = self._calculate_max_pooling_scores(all_individual_results, top_k)
            avg_results = self._calculate_avg_pooling_scores(all_individual_results, top_k)
            
            logger.info(f"Multi-score image search: {len(max_results)} max events, {len(avg_results)} avg events")
            return max_results, avg_results
            
        except Exception as e:
            logger.error(f"Error in multi-score image search: {e}")
            raise

    def _calculate_max_pooling_scores(
        self, 
        all_results: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Calculate max pooling scores for events"""
        event_groups = defaultdict(list)
        
        # Group by FAISS index (representing unique images)
        for result in all_results:
            faiss_idx = result['index']
            event_groups[faiss_idx].append(result['score'])
        
        # Calculate max scores
        max_scores = []
        for faiss_idx, scores in event_groups.items():
            max_score = max(scores)
            max_scores.append({
                'index': faiss_idx,
                'score': max_score,
                'pooling_type': 'max',
                'match_count': len(scores)
            })
        
        # Sort by score and return top_k
        max_scores.sort(key=lambda x: x['score'], reverse=True)
        return max_scores[:top_k]

    def _calculate_avg_pooling_scores(
        self, 
        all_results: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Calculate average pooling scores for events"""
        event_groups = defaultdict(list)
        
        # Group by FAISS index
        for result in all_results:
            faiss_idx = result['index']
            event_groups[faiss_idx].append(result['score'])
        
        # Calculate average scores
        avg_scores = []
        for faiss_idx, scores in event_groups.items():
            avg_score = sum(scores) / len(scores)
            avg_scores.append({
                'index': faiss_idx,
                'score': avg_score,
                'pooling_type': 'avg',
                'match_count': len(scores)
            })
        
        # Sort by score and return top_k
        avg_scores.sort(key=lambda x: x['score'], reverse=True)
        return avg_scores[:top_k]

    def _combine_text_results_only(
        self, 
        name_results: List[Dict], 
        about_results: List[Dict]
    ) -> List[int]:
        """NEW: Combine only text search results (name + about)"""
        text_faiss_indices = set()
        
        for result in name_results:
            if 'index' in result:
                text_faiss_indices.add(result['index'])
        
        for result in about_results:
            if 'index' in result:
                text_faiss_indices.add(result['index'])
        
        indices_list = list(text_faiss_indices)
        logger.info(f"Combined text results: {len(name_results)} name + {len(about_results)} about = {len(indices_list)} unique")
        
        return indices_list

    async def _process_text_events_with_claude(
        self,
        query_name: str,
        query_about: str,
        game_code: str,
        text_faiss_indices: List[int]
    ) -> List[Dict[str, Any]]:
        """NEW: Process text events through Claude analysis"""
        try:
            # Get text data from database
            candidate_texts = await self.query_postgres_for_texts(text_faiss_indices)
            
            if not candidate_texts:
                logger.warning("No text candidates found for Claude analysis")
                return []
            
            # Send to Claude for analysis
            claude_analysis = await self.analyze_text_similarity_with_claude(
                query_name, query_about, game_code, candidate_texts
            )
            
            # Extract processed candidates with scores
            claude_candidates = claude_analysis.get("formatted_candidates", []) if claude_analysis else []
            
            # Add text-only flags
            for candidate in claude_candidates:
                candidate['found_by'] = 'text'
                candidate['score_image_max'] = 0.0
                candidate['score_image_avg'] = 0.0
            
            logger.info(f"Claude processed {len(claude_candidates)} text events")
            return claude_candidates
            
        except Exception as e:
            logger.error(f"Error processing text events with Claude: {e}")
            return []

    def _merge_text_and_image_events(
        self,
        text_events: List[Dict[str, Any]],
        image_events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """NEW: Merge text and image events with proper score handling"""
        
        # Create lookup for text events
        text_events_lookup = {event['event_code']: event for event in text_events}
        
        # Create lookup for image events  
        image_events_lookup = {event['event_code']: event for event in image_events}
        
        # Get all unique event codes
        all_event_codes = set(text_events_lookup.keys()) | set(image_events_lookup.keys())
        
        merged_events = []
        
        for event_code in all_event_codes:
            text_event = text_events_lookup.get(event_code)
            image_event = image_events_lookup.get(event_code)
            
            if text_event and image_event:
                # Overlapping event: merge scores
                merged_event = text_event.copy()
                merged_event['score_image_max'] = image_event['score_image_max']
                merged_event['score_image_avg'] = image_event['score_image_avg']
                merged_event['found_by'] = 'both'
                merged_event['reason'] = f"{text_event['reason']} Also found by visual similarity."
                
            elif text_event:
                # Text-only event: already has image scores = 0.0
                merged_event = text_event.copy()
                
            else:
                # Image-only event: already has text score = 0.0
                merged_event = image_event.copy()
            
            merged_events.append(merged_event)
        
        logger.info(f"Merged events: {len(text_events)} text + {len(image_events)} image = {len(merged_events)} final")
        return merged_events  


    def _build_multi_score_response(
        self,
        query_name: str,
        query_about: str, 
        merged_events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """NEW: Build response with 3-score system"""
        
        # Format similar events for response
        similar_events = []
        for event in merged_events:
            similar_event = {
                "name": event["event_name"],
                "about": event.get("about_content", ""),
                "score_text": event["score_text"],
                "score_image_max": event["score_image_max"],
                "score_image_avg": event["score_image_avg"],
                "reason": event["reason"],
                "tags": event["taxonomy"],
                "tag_explanation": event["tag_explanation"]
            }
            similar_events.append(similar_event)
        
        # Query event taxonomy (use first text event's analysis if available)
        query_tags = {"family": "Other", "dynamics": "Individualistic", "rewards": "None"}
        query_explanation = "Phase-separated analysis - query event taxonomy not analyzed"
        
        # Build final response
        response = {
            "query_event": {
                "name": query_name,
                "about": query_about,
                "tags": query_tags,
                "tag_explanation": query_explanation
            },
            "similar_events": similar_events
        }
        
        return response   

    async def _process_image_events_separately(
        self,
        image_results_max: List[Dict[str, Any]],
        image_results_avg: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process image events separately with direct FAISS score mapping"""
        try:
            # Step 1: Get all unique FAISS indices from search results
            all_image_indices = set()
            for result in image_results_max:
                all_image_indices.add(result['index'])
            for result in image_results_avg:
                all_image_indices.add(result['index'])
            
            if not all_image_indices:
                logger.warning("No image FAISS indices found")
                return []
            
            # Step 2: Create score lookups from search results
            max_scores = {r['index']: r['score'] for r in image_results_max}
            avg_scores = {r['index']: r['score'] for r in image_results_avg}
            
            # Step 3: Query database to get event data by FAISS indices
            # This uses the corrected database query that goes: images → events → text_embeddings
            image_candidates = await self.query_postgres_for_texts(list(all_image_indices))
            
            if not image_candidates:
                logger.warning(f"No events found for FAISS indices: {list(all_image_indices)[:10]}...")
                return []
            
            # Step 4: Create processed events with direct score assignment
            processed_image_events = []
            
            # Note: We need to map back from event to its source FAISS indices
            # Since multiple FAISS indices might point to same event, we need to find the best scores
            
            # Get mapping from event_code to source FAISS indices
            event_faiss_mapping = await self._get_source_faiss_indices_for_events(
                [c['event_code'] for c in image_candidates]
            )
            
            for candidate in image_candidates:
                event_code = candidate['event_code']
                
                # Find all FAISS indices that led to this event being found
                source_indices = event_faiss_mapping.get(event_code, [])
                
                # Calculate best scores from all source indices
                event_max_scores = [max_scores.get(idx, 0.0) for idx in source_indices if idx in max_scores]
                event_avg_scores = [avg_scores.get(idx, 0.0) for idx in source_indices if idx in avg_scores]
                
                # Use the best scores (highest) for this event
                final_max_score = max(event_max_scores) if event_max_scores else 0.0
                final_avg_score = max(event_avg_scores) if event_avg_scores else 0.0
                
                # Build the processed event
                image_event = {
                    'event_code': event_code,
                    'event_name': candidate['event_name'],
                    'about_content': candidate['about_content'],
                    'found_by': 'image',
                    'score_text': 0.0,
                    'score_image_max': final_max_score,
                    'score_image_avg': final_avg_score,
                    'reason': f"Found by visual similarity (max: {final_max_score:.3f}, avg: {final_avg_score:.3f})",
                    'taxonomy': {
                        'family': 'Other',
                        'dynamics': 'Individualistic', 
                        'rewards': 'None'
                    },
                    'tag_explanation': 'Image-found event - text analysis not performed',
                    'source_faiss_indices': source_indices  # For debugging
                }
                
                processed_image_events.append(image_event)
            
            logger.info(f"Processed {len(processed_image_events)} image events with direct score mapping")
            return processed_image_events
            
        except Exception as e:
            logger.error(f"Error processing image events: {e}")
            return []

    async def _get_source_faiss_indices_for_events(self, event_codes: List[str]) -> Dict[str, List[int]]:
        """
        Get the FAISS indices that correspond to each event_code
        This is needed because one event can have multiple images (multiple FAISS indices)
        """
        try:
            if not event_codes:
                return {}
            
            # Query the images table to find which FAISS indices belong to each event
            query = """
            SELECT i.event_code, i.faiss_index 
            FROM images i
            WHERE i.event_code = ANY($1::uuid[])
            AND i.is_deleted = false
            AND i.faiss_index IS NOT NULL
            """
            
            results = await self.database_service.execute_query(query, [event_codes])
            
            # Group FAISS indices by event_code
            mapping = {}
            for row in results:
                event_code = row['event_code']
                faiss_index = row['faiss_index']
                
                if event_code not in mapping:
                    mapping[event_code] = []
                mapping[event_code].append(faiss_index)
            
            logger.debug(f"Source FAISS mapping: {len(mapping)} events mapped to their source indices")
            return mapping
            
        except Exception as e:
            logger.error(f"Error getting source FAISS indices for events: {e}")
            return {}           