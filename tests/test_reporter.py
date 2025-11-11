"""
Test Reporter Module

Handles displaying and saving test results.
"""

import json
import asyncio
from typing import Dict, Any


def display_results(result: Dict[str, Any], execution_time: float, verbose: bool = True) -> None:
    """
    Display test results in a formatted way

    Args:
        result: Test results dictionary
        execution_time: Time taken to execute the test
        verbose: Whether to show verbose output
    """
    print("\n📊 TEST RESULTS")
    print("=" * 60)

    # Query event information
    query_event = result.get('query_event', {})
    print("🎯 QUERY EVENT:")
    print(f"   Name: {query_event.get('name', 'N/A')}")
    print(f"   About: {query_event.get('about', 'N/A')[:100]}...")

    tags = query_event.get('tags', {})
    print(f"   Tags:")
    print(f"     Family: {tags.get('family', 'N/A')}")
    print(f"     Dynamics: {tags.get('dynamics', 'N/A')}")
    print(f"     Rewards: {tags.get('rewards', 'N/A')}")

    if verbose:
        print(f"   Tag Explanation: {query_event.get('tag_explanation', 'N/A')}")

    # Similar events
    similar_events = result.get('similar_events', [])
    print(f"\n🔍 SIMILAR EVENTS FOUND: {len(similar_events)}")

    if similar_events:
        print("\nTop Similar Events:")
        for i, event in enumerate(similar_events[:5], 1):  # Show top 5
            print(f"\n{i}. {event.get('name', 'N/A')}")
            print(f"   Text Score: {event.get('score_text', 0.0):.3f}")
            print(f"   Image Score: {event.get('score_image', 0.0):.3f}")
            print(f"   Matching Images: {len(event.get('image_faiss_indices', []))} indices")

            event_tags = event.get('tags', {})
            print(f"   Tags: {event_tags.get('family', 'N/A')} | {event_tags.get('dynamics', 'N/A')} | {event_tags.get('rewards', 'N/A')}")

            if verbose:
                print(f"   Reason: {event.get('reason', 'N/A')[:150]}...")
                print(f"   About: {event.get('about', 'N/A')[:100]}...")

        # Statistics
        text_scores = [e.get('score_text', 0.0) for e in similar_events]
        image_scores = [e.get('score_image', 0.0) for e in similar_events]

        print(f"\n📈 STATISTICS:")
        print(f"   Total Events Found: {len(similar_events)}")
        print(f"   Average Text Score: {sum(text_scores)/len(text_scores):.3f}")
        print(f"   Average Image Score: {sum(image_scores)/len(image_scores):.3f}")
        print(f"   Max Text Score: {max(text_scores):.3f}")
        print(f"   Max Image Score: {max(image_scores):.3f}")
        print(f"   Execution Time: {execution_time:.2f} seconds")
    else:
        print("   No similar events found")


def save_test_results(
    test_request: Dict[str, Any],
    result: Dict[str, Any],
    execution_time: float,
    output_file: str = "test_results.json",
    config: Any = None
) -> None:
    """
    Save test results to JSON file

    Args:
        test_request: Test request parameters
        result: Test results
        execution_time: Execution time in seconds
        output_file: Output file name
        config: Test configuration object
    """
    try:
        import time

        output_data = {
            "test_configuration": {
                "timestamp": time.time(),
                "execution_time_seconds": execution_time,
                "request_parameters": test_request,
            },
            "results": result
        }

        # Add config details if provided
        if config:
            output_data["test_configuration"]["config"] = {
                "event_name": getattr(config, 'EVENT_NAME', None),
                "game_code": getattr(config, 'GAME_CODE', None),
                "folder_name": getattr(config, 'FOLDER_NAME', None),
                "expected_image_count": getattr(config, 'EXPECTED_IMAGE_COUNT', None)
            }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to: {output_file}")

    except Exception as e:
        print(f"⚠️ Failed to save results: {e}")
