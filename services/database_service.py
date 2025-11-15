"""
Database Service

Handles PostgreSQL database connections and queries for event similarity search.
"""

import logging
from typing import List, Dict, Any, Optional
import asyncpg
from datetime import datetime
from config import config

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service for handling PostgreSQL database operations"""

    def __init__(self):
        self.pool = None
        self.db_config = config.get_db_config()
    
    async def initialize(self) -> bool:
        """Initialize PostgreSQL connection pool"""
        try:
            # Validate required environment variables
            missing_vars = []
            for key, value in self.db_config.items():
                if value is None:
                    missing_vars.append(f"DB_{key.upper()}")
            
            if missing_vars:
                logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
                return False
            
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                host=self.db_config['host'],
                port=self.db_config['port'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database'],
                min_size=2,
                max_size=10,
                command_timeout=60,
                server_settings={
                    'application_name': 'ai_service_event_similarity'
                }
            )
            
            # Test connection
            async with self.pool.acquire() as conn:
                await conn.execute('SELECT 1')
            
            logger.info(f"Database connection pool initialized successfully")
            logger.info(f"Connected to: {self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            return False
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
    
    async def get_events_by_codes(self, event_codes: List[str]) -> List[Dict[str, Any]]:
        """
        Get event details by event codes
        
        Args:
            event_codes: List of event code UUIDs
            
        Returns:
            List of event dictionaries
        """
        if not event_codes:
            return []
        
        if not self.pool:
            logger.error("Database connection pool not initialized")
            return []
        
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT 
                        id,
                        code,
                        game_code,
                        name,
                        search_key,
                        author,
                        publish_date,
                        created_at
                    FROM events 
                    WHERE code = ANY($1::uuid[])
                    ORDER BY created_at DESC
                """
                
                rows = await conn.fetch(query, event_codes)
                
                results = []
                for row in rows:
                    results.append({
                        "id": row["id"],
                        "code": str(row["code"]),
                        "game_code": row["game_code"],
                        "name": row["name"],
                        "search_key": row["search_key"],
                        "author": row["author"],
                        "publish_date": row["publish_date"].isoformat() if row["publish_date"] else None,
                        "created_at": row["created_at"].isoformat() if row["created_at"] else None
                    })
                
                logger.info(f"Retrieved {len(results)} events from database")
                return results
                
        except Exception as e:
            logger.error(f"Error querying events by codes: {e}")
            return []
    
    async def get_text_embeddings_by_event_codes(
        self, 
        event_codes: List[str], 
        content_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get text embeddings by event codes and content types
        
        Args:
            event_codes: List of event code UUIDs
            content_types: List of content types to filter ('name', 'about')
            
        Returns:
            List of text embedding dictionaries
        """
        if not event_codes:
            return []
        
        if not self.pool:
            logger.error("Database connection pool not initialized")
            return []
        
        try:
            async with self.pool.acquire() as conn:
                # Build query with optional content type filter
                base_query = """
                    SELECT 
                        id,
                        code,
                        event_code,
                        content_type,
                        text_content,
                        text_length,
                        faiss_index,
                        vector_dimension,
                        created_at
                    FROM text_embeddings 
                    WHERE event_code = ANY($1::uuid[])
                """
                
                params = [event_codes]
                if content_types:
                    base_query += " AND content_type = ANY($2::varchar[])"
                    params.append(content_types)
                
                base_query += " ORDER BY created_at DESC"
                
                rows = await conn.fetch(base_query, *params)
                
                results = []
                for row in rows:
                    results.append({
                        "id": row["id"],
                        "code": str(row["code"]),
                        "event_code": str(row["event_code"]),
                        "content_type": row["content_type"],
                        "text_content": row["text_content"],
                        "text_length": row["text_length"],
                        "faiss_index": row["faiss_index"],
                        "vector_dimension": row["vector_dimension"],
                        "created_at": row["created_at"].isoformat() if row["created_at"] else None
                    })
                
                logger.info(f"Retrieved {len(results)} text embeddings from database")
                return results
                
        except Exception as e:
            logger.error(f"Error querying text embeddings by event codes: {e}")
            return []
    
    async def get_events_by_faiss_indices(self, faiss_indices: List[int]) -> List[Dict[str, Any]]:
        """
        Get event data by FAISS indices from text_embeddings table
        
        Args:
            faiss_indices: List of FAISS index integers
            
        Returns:
            List of combined event data with text embeddings
        """
        if not faiss_indices:
            return []
        
        if not self.pool:
            logger.error("Database connection pool not initialized")
            return []
        
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT 
                        e.id as event_id,
                        e.code as event_code,
                        e.game_code,
                        e.name as event_name,
                        e.search_key,
                        e.author,
                        e.publish_date,
                        e.created_at as event_created_at,
                        te.id as embedding_id,
                        te.content_type,
                        te.text_content,
                        te.text_length,
                        te.faiss_index,
                        te.created_at as embedding_created_at
                    FROM text_embeddings te
                    JOIN events e ON e.code = te.event_code
                    WHERE te.faiss_index = ANY($1::int[])
                    ORDER BY e.created_at DESC, te.content_type
                """
                
                rows = await conn.fetch(query, faiss_indices)
                
                # Group by event_code
                events_dict = {}
                for row in rows:
                    event_code = str(row["event_code"])
                    
                    if event_code not in events_dict:
                        events_dict[event_code] = {
                            "event_id": row["event_id"],
                            "event_code": event_code,
                            "game_code": row["game_code"],
                            "event_name": row["event_name"],
                            "search_key": row["search_key"],
                            "author": row["author"],
                            "publish_date": row["publish_date"].isoformat() if row["publish_date"] else None,
                            "event_created_at": row["event_created_at"].isoformat() if row["event_created_at"] else None,
                            "text_embeddings": {}
                        }
                    
                    # Add text embedding
                    content_type = row["content_type"]
                    events_dict[event_code]["text_embeddings"][content_type] = {
                        "embedding_id": row["embedding_id"],
                        "text_content": row["text_content"],
                        "text_length": row["text_length"],
                        "faiss_index": row["faiss_index"],
                        "embedding_created_at": row["embedding_created_at"].isoformat() if row["embedding_created_at"] else None
                    }
                
                results = list(events_dict.values())
                logger.info(f"Retrieved {len(results)} events by FAISS indices from database")
                return results
                
        except Exception as e:
            logger.error(f"Error querying events by FAISS indices: {e}")
            return []
    
    async def get_all_events_for_game(self, game_code: str) -> List[Dict[str, Any]]:
        """
        Get all events for a specific game

        Args:
            game_code: Game code (e.g., 'candy_crush')

        Returns:
            List of event dictionaries with event_code
        """
        if not self.pool:
            logger.error("Database connection pool not initialized")
            return []

        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT
                        code as event_code,
                        name as event_name,
                        game_code
                    FROM events
                    WHERE game_code = $1
                    ORDER BY created_at DESC
                """

                rows = await conn.fetch(query, game_code)

                results = []
                for row in rows:
                    results.append({
                        "event_code": str(row["event_code"]),
                        "event_name": row["event_name"],
                        "game_code": row["game_code"]
                    })

                logger.info(f"Retrieved {len(results)} events for game {game_code}")
                return results

        except Exception as e:
            logger.error(f"Error querying events for game {game_code}: {e}")
            return []

    async def get_image_faiss_indices_for_event(self, event_code: str, game_code: str) -> List[int]:
        """
        Get all image FAISS indices for a specific event

        This assumes image_embeddings table exists with:
        - event_code (UUID)
        - faiss_index (INTEGER)
        - file_name (TEXT)

        Args:
            event_code: Event code UUID
            game_code: Game code (for future filtering if needed)

        Returns:
            List of FAISS index integers
        """
        if not self.pool:
            logger.error("Database connection pool not initialized")
            return []

        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT faiss_index
                    FROM image_embeddings
                    WHERE event_code = $1
                    ORDER BY faiss_index
                """

                rows = await conn.fetch(query, event_code)

                indices = [int(row["faiss_index"]) for row in rows]
                logger.debug(f"Retrieved {len(indices)} image FAISS indices for event {event_code}")
                return indices

        except Exception as e:
            logger.error(f"Error querying image FAISS indices for event {event_code}: {e}")
            return []

    async def get_images_for_events(self, event_codes: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all images for multiple events

        Args:
            event_codes: List of event code UUIDs

        Returns:
            Dict mapping event_code to list of image info dicts
        """
        if not event_codes:
            return {}

        if not self.pool:
            logger.error("Database connection pool not initialized")
            return {}

        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT
                        event_code,
                        faiss_index,
                        file_name,
                        created_at
                    FROM image_embeddings
                    WHERE event_code = ANY($1::uuid[])
                    ORDER BY event_code, faiss_index
                """

                rows = await conn.fetch(query, event_codes)

                # Group by event_code
                results = {}
                for row in rows:
                    event_code = str(row["event_code"])
                    if event_code not in results:
                        results[event_code] = []

                    results[event_code].append({
                        "faiss_index": row["faiss_index"],
                        "file_name": row["file_name"],
                        "created_at": row["created_at"].isoformat() if row["created_at"] else None
                    })

                logger.info(f"Retrieved images for {len(results)} events")
                return results

        except Exception as e:
            logger.error(f"Error querying images for events: {e}")
            return {}

    async def health_check(self) -> Dict[str, Any]:
        """Check database connection health"""
        if not self.pool:
            return {
                "status": "error",
                "message": "Database connection pool not initialized",
                "timestamp": datetime.utcnow().isoformat()
            }

        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval('SELECT COUNT(*) FROM events')

                return {
                    "status": "healthy",
                    "total_events": result,
                    "pool_size": self.pool.get_size(),
                    "pool_free_size": self.pool.get_idle_size(),
                    "config": {
                        "host": self.db_config['host'],
                        "port": self.db_config['port'],
                        "database": self.db_config['database']
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        
    async def execute_query(self, query: str, params = None) -> List[Dict[str, Any]]:
        """
        Execute raw SQL query and return results
        
        Args:
            query: SQL query string
            params: Query parameters (tuple, list, or single value)
            
        Returns:
            List of dictionaries representing query results
        """
        if not self.pool:
            logger.error("Database connection pool not initialized")
            return []
        
        try:
            async with self.pool.acquire() as conn:
                if params is not None:
                    # FIXED: Handle multiple parameter types
                    if isinstance(params, (list, tuple)):
                        # Multiple parameters - unpack with *
                        rows = await conn.fetch(query, *params)
                    else:
                        # Single parameter - pass directly
                        rows = await conn.fetch(query, params)
                else:
                    rows = await conn.fetch(query)
                
                results = []
                for row in rows:
                    # Convert asyncpg.Record to dict
                    result_dict = dict(row)
                    
                    # Convert UUID objects to strings
                    for key, value in result_dict.items():
                        if hasattr(value, '__class__') and 'UUID' in str(value.__class__):
                            result_dict[key] = str(value)
                        elif hasattr(value, 'isoformat'):  # datetime objects
                            result_dict[key] = value.isoformat()
                    
                    results.append(result_dict)
                
                logger.debug(f"Executed query, returned {len(results)} rows")
                return results
                
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            return []