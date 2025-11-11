"""
Test Runner Module

Handles execution of event similarity tests.
"""

import asyncio
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


async def run_event_similarity_test(
    services: Dict[str, Any],
    test_request: Dict[str, Any],
    verbose: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Run the complete event similarity analysis test

    Args:
        services: Dictionary of initialized services
        test_request: Test request parameters
        verbose: Whether to print verbose output

    Returns:
        Test results dictionary or None if failed
    """
    if verbose:
        print("\n🚀 Starting Event Similarity Analysis Test...")
        print("=" * 60)

    if not services.get('event_similarity_service'):
        if verbose:
            print("❌ Event Similarity service not available")
        return None

    if verbose:
        print("📋 Test Parameters:")
        print(f"   Folder: {test_request['folder_name']}")
        print(f"   Game: {test_request['game_code']}")
        print(f"   Event: {test_request['event_name']}")
        print(f"   Images: {test_request['image_count']}")
        print(f"   Path: {test_request['shared_uploads_path']}")
        print()

    try:
        # Run the analysis
        if verbose:
            print("🔄 Running similarity analysis...")
        start_time = asyncio.get_event_loop().time()

        result = await services['event_similarity_service'].find_similar_events(
            query_name=test_request['event_name'],
            query_about=test_request['about'],
            folder_name=test_request['folder_name'],
            game_code=test_request['game_code'],
            shared_uploads_path=test_request['shared_uploads_path'],
            image_count=test_request['image_count']
        )

        end_time = asyncio.get_event_loop().time()
        execution_time = end_time - start_time

        if verbose:
            print(f"✅ Analysis completed in {execution_time:.2f} seconds")

        return {
            'result': result,
            'execution_time': execution_time
        }

    except Exception as e:
        if verbose:
            print(f"❌ Test failed: {e}")
        logger.exception("Detailed error information:")
        return None
