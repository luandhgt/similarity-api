"""
Event Repository

Handles all database operations related to events.
Provides abstraction over raw SQL queries.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from core.exceptions import DatabaseError, NotFoundError

logger = logging.getLogger(__name__)


class EventRepository:
    """Repository for event data access"""

    def __init__(self, database_service):
        """
        Initialize repository with database service

        Args:
            database_service: DatabaseService instance
        """
        self.db = database_service

    async def find_by_game_code(
        self,
        game_code: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Find events by game code

        Args:
            game_code: Game identifier
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of event dictionaries

        Raises:
            DatabaseError: If query fails
        """
        try:
            query = """
                SELECT *
                FROM events
                WHERE game_code = $1
                ORDER BY created_at DESC
            """

            if limit:
                query += f" LIMIT {limit} OFFSET {offset}"

            async with self.db.pool.acquire() as conn:
                rows = await conn.fetch(query, game_code)

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to find events by game_code '{game_code}': {e}")
            raise DatabaseError(
                f"Failed to query events",
                operation="find_by_game_code",
                details={'game_code': game_code, 'error': str(e)}
            )

    async def find_by_faiss_indices(
        self,
        faiss_indices: List[int],
        game_code: str
    ) -> List[Dict[str, Any]]:
        """
        Find events by their FAISS indices

        Args:
            faiss_indices: List of FAISS index IDs
            game_code: Game identifier

        Returns:
            List of event dictionaries

        Raises:
            DatabaseError: If query fails
        """
        if not faiss_indices:
            return []

        try:
            query = """
                SELECT *
                FROM events
                WHERE game_code = $1
                  AND faiss_index = ANY($2::int[])
            """

            async with self.db.pool.acquire() as conn:
                rows = await conn.fetch(query, game_code, faiss_indices)

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to find events by FAISS indices: {e}")
            raise DatabaseError(
                f"Failed to query events by FAISS indices",
                operation="find_by_faiss_indices",
                details={'game_code': game_code, 'indices_count': len(faiss_indices), 'error': str(e)}
            )

    async def find_similar_by_name(
        self,
        name: str,
        game_code: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find events with similar names using text search

        Args:
            name: Event name to search for
            game_code: Game identifier
            limit: Maximum number of results

        Returns:
            List of event dictionaries

        Raises:
            DatabaseError: If query fails
        """
        try:
            query = """
                SELECT *
                FROM events
                WHERE game_code = $1
                  AND name ILIKE $2
                ORDER BY similarity(name, $3) DESC
                LIMIT $4
            """

            search_pattern = f"%{name}%"

            async with self.db.pool.acquire() as conn:
                rows = await conn.fetch(query, game_code, search_pattern, name, limit)

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to find similar events by name: {e}")
            raise DatabaseError(
                f"Failed to search events by name",
                operation="find_similar_by_name",
                details={'name': name, 'game_code': game_code, 'error': str(e)}
            )

    async def get_by_id(self, event_id: int) -> Dict[str, Any]:
        """
        Get event by ID

        Args:
            event_id: Event ID

        Returns:
            Event dictionary

        Raises:
            NotFoundError: If event not found
            DatabaseError: If query fails
        """
        try:
            query = "SELECT * FROM events WHERE id = $1"

            async with self.db.pool.acquire() as conn:
                row = await conn.fetchrow(query, event_id)

            if not row:
                raise NotFoundError("Event", str(event_id))

            return dict(row)

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get event by ID {event_id}: {e}")
            raise DatabaseError(
                f"Failed to get event",
                operation="get_by_id",
                details={'event_id': event_id, 'error': str(e)}
            )

    async def create(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new event

        Args:
            event_data: Event data dictionary

        Returns:
            Created event dictionary with ID

        Raises:
            DatabaseError: If insert fails
        """
        try:
            query = """
                INSERT INTO events (
                    name, about, game_code, tags,
                    faiss_index_name, faiss_index_about, faiss_index_images,
                    created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING *
            """

            async with self.db.pool.acquire() as conn:
                row = await conn.fetchrow(
                    query,
                    event_data.get('name'),
                    event_data.get('about'),
                    event_data.get('game_code'),
                    event_data.get('tags'),
                    event_data.get('faiss_index_name'),
                    event_data.get('faiss_index_about'),
                    event_data.get('faiss_index_images'),
                    event_data.get('created_at', datetime.now())
                )

            return dict(row)

        except Exception as e:
            logger.error(f"Failed to create event: {e}")
            raise DatabaseError(
                f"Failed to insert event",
                operation="create",
                details={'event_data': event_data, 'error': str(e)}
            )

    async def update(self, event_id: int, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing event

        Args:
            event_id: Event ID
            event_data: Event data to update

        Returns:
            Updated event dictionary

        Raises:
            NotFoundError: If event not found
            DatabaseError: If update fails
        """
        try:
            # Build dynamic update query
            fields = []
            values = []
            param_idx = 1

            for key, value in event_data.items():
                if key != 'id':  # Don't update ID
                    fields.append(f"{key} = ${param_idx}")
                    values.append(value)
                    param_idx += 1

            if not fields:
                raise ValueError("No fields to update")

            values.append(event_id)  # Add ID as last parameter

            query = f"""
                UPDATE events
                SET {', '.join(fields)}, updated_at = NOW()
                WHERE id = ${param_idx}
                RETURNING *
            """

            async with self.db.pool.acquire() as conn:
                row = await conn.fetchrow(query, *values)

            if not row:
                raise NotFoundError("Event", str(event_id))

            return dict(row)

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update event {event_id}: {e}")
            raise DatabaseError(
                f"Failed to update event",
                operation="update",
                details={'event_id': event_id, 'error': str(e)}
            )

    async def delete(self, event_id: int) -> bool:
        """
        Delete an event

        Args:
            event_id: Event ID

        Returns:
            True if deleted

        Raises:
            NotFoundError: If event not found
            DatabaseError: If delete fails
        """
        try:
            query = "DELETE FROM events WHERE id = $1 RETURNING id"

            async with self.db.pool.acquire() as conn:
                row = await conn.fetchrow(query, event_id)

            if not row:
                raise NotFoundError("Event", str(event_id))

            return True

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete event {event_id}: {e}")
            raise DatabaseError(
                f"Failed to delete event",
                operation="delete",
                details={'event_id': event_id, 'error': str(e)}
            )

    async def count_by_game(self, game_code: str) -> int:
        """
        Count events for a game

        Args:
            game_code: Game identifier

        Returns:
            Number of events

        Raises:
            DatabaseError: If query fails
        """
        try:
            query = "SELECT COUNT(*) FROM events WHERE game_code = $1"

            async with self.db.pool.acquire() as conn:
                count = await conn.fetchval(query, game_code)

            return count or 0

        except Exception as e:
            logger.error(f"Failed to count events for game '{game_code}': {e}")
            raise DatabaseError(
                f"Failed to count events",
                operation="count_by_game",
                details={'game_code': game_code, 'error': str(e)}
            )


__all__ = ['EventRepository']
