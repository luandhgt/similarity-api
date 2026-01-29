"""
Compare embedding providers: Voyage vs OpenAI

Usage:
    # Compare with Voyage (default)
    python debug_compare_embeddings.py

    # Compare with OpenAI
    EMBEDDING_PROVIDER=openai python debug_compare_embeddings.py

    # Compare both
    python debug_compare_embeddings.py --compare-both
"""

import asyncio
import numpy as np
from pathlib import Path
import sys
import os
import argparse
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
env_file = Path(__file__).parent / ".env.development"
if env_file.exists():
    load_dotenv(env_file)
    print(f"Loaded env from {env_file}")
else:
    load_dotenv()
    print("Loaded env from .env")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


# =============================================================================
# TEST TEXTS
# =============================================================================

# =============================================================================
# MECHANIC-FOCUSED EVENT DESCRIPTIONS (rewritten to emphasize gameplay mechanics)
# =============================================================================

# QUERY - Weathered Brilliance (Mechanic-focused)
QUERY_TEXT_LONG = '''"Weathered Brilliance" is a limited-time challenge event where players progress through a series of sequential stages on a map interface. Each stage must be completed before advancing to the next. The event features two difficulty tiers: Normal Mode stages that must be cleared first, followed by Hard Mode stages that unlock after completing all standard levels.

Players are ranked on a kingdom-wide leaderboard based on how fast they complete all stages. The ranking system separates results achieved with and without item use, creating multiple competitive tiers. Each stage can be replayed to improve completion time.

Rewards are earned progressively: players receive items and speedup boosters upon completing individual stages, with additional milestone rewards for overall progress. The event structure is individualistic with competitive ranking elements.'''

# SHORT QUERIES - Mechanic-focused queries
SHORT_QUERIES = {
    "stage_progression": "event with sequential stages that must be completed in order",
    "speed_ranking": "players ranked by how fast they complete all stages",
    "hard_mode_unlock": "complete normal stages to unlock hard mode for additional challenges",
    "milestone_rewards": "earn rewards upon completing stages with milestone bonuses",
}

# EVENT 1 - War of Ruins (Mechanic-focused)
EVENT1_TEXT = '''"War of the Ruins" is a solo competitive event where players engage in real-time PvP battles against other players. Players select and control troops with various skills on a battlefield, with the objective to defeat opponents and dominate the arena.

Progress is measured through Victory Points earned during battles. Players accumulate Victory Points by achieving combat milestones, and these points determine both ranking position and reward tiers. The event uses matchmaking to pair players of similar skill levels.

Rewards are primarily based on final ranking position determined by total Victory Points. There is no stage-based progression - instead, players continuously battle to accumulate points throughout the event duration.'''

# EVENT 2 - All Under Heaven (Mechanic-focused)
EVENT2_TEXT = '''"All Under Heaven" is a strategic challenge event where players progress through sequential stages by conquering territories on a campaign map. Each stage requires players to occupy all enemy regions to advance. Players complete stages in order, with each campaign offering multiple stages of increasing difficulty.

After clearing all standard stages, players unlock harder challenge modes for additional progression. The event ranks players based on the speed of stage completion, creating a competitive leaderboard for fastest completers.

Rewards are earned upon completing each stage, with milestone rewards at key progression points. Additional rewards are available for clearing hard mode stages. The event structure is individualistic with competitive speed-based ranking.'''

# Default query (for backward compatibility)
QUERY_TEXT = QUERY_TEXT_LONG


def test_provider(provider_name: str, provider, test_short_queries: bool = True):
    """Test a single embedding provider"""
    print(f"\n{'=' * 80}")
    print(f"Testing {provider_name.upper()} Embedding Provider")
    print(f"{'=' * 80}")

    info = provider.get_provider_info()
    print(f"Model: {info.get('model')}")
    print(f"Dimensions: {info.get('dimensions')}")

    # Generate embeddings for events (only once)
    print("\nGenerating event embeddings...")
    event1_vec = provider.embed_text(EVENT1_TEXT)
    event2_vec = provider.embed_text(EVENT2_TEXT)
    print(f"  Event 1 vector shape: {event1_vec.shape}")
    print(f"  Event 2 vector shape: {event2_vec.shape}")

    results = []

    # Test 1: Long query (original)
    print(f"\n--- Test 1: LONG QUERY (full event description) ---")
    query_vec = provider.embed_text(QUERY_TEXT_LONG)
    sim_query_event1 = cosine_similarity(query_vec, event1_vec)
    sim_query_event2 = cosine_similarity(query_vec, event2_vec)

    print(f"  Query vs Event 1 (War of Ruins): {sim_query_event1:.4f}")
    print(f"  Query vs Event 2 (All Under Heaven): {sim_query_event2:.4f}")
    winner = "Event 2" if sim_query_event2 > sim_query_event1 else "Event 1"
    status = "✅" if winner == "Event 2" else "❌"
    print(f"  Winner: {winner} {status} (diff: {abs(sim_query_event1 - sim_query_event2):.4f})")

    results.append({
        "query_type": "long",
        "sim_event1": sim_query_event1,
        "sim_event2": sim_query_event2,
        "winner": winner,
    })

    # Test 2: Short queries (realistic user searches)
    if test_short_queries:
        print(f"\n--- Test 2: SHORT QUERIES (realistic user searches) ---")
        for query_name, query_text in SHORT_QUERIES.items():
            query_vec = provider.embed_text(query_text)
            sim_event1 = cosine_similarity(query_vec, event1_vec)
            sim_event2 = cosine_similarity(query_vec, event2_vec)
            winner = "Event 2" if sim_event2 > sim_event1 else "Event 1"
            status = "✅" if winner == "Event 2" else "❌"

            print(f"\n  [{query_name}] \"{query_text[:50]}...\"")
            print(f"    Event 1: {sim_event1:.4f} | Event 2: {sim_event2:.4f} | Winner: {winner} {status}")

            results.append({
                "query_type": query_name,
                "query_text": query_text,
                "sim_event1": sim_event1,
                "sim_event2": sim_event2,
                "winner": winner,
            })

    # Summary
    correct_count = sum(1 for r in results if r["winner"] == "Event 2")
    total_count = len(results)

    print(f"\n📊 SUMMARY: {correct_count}/{total_count} queries correctly ranked Event 2 higher")

    return {
        "provider": provider_name,
        "model": info.get("model"),
        "dimensions": info.get("dimensions"),
        "results": results,
        "correct_rate": correct_count / total_count if total_count > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare embedding providers")
    parser.add_argument(
        "--compare-both",
        action="store_true",
        help="Compare both Voyage and OpenAI providers"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Specific provider to test (voyage, openai)"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("EMBEDDING PROVIDERS COMPARISON")
    print("=" * 80)
    print("\nTest setup:")
    print("  Event 1: 'War of the Ruins' - PvP combat, Victory Points, battlefield")
    print("  Event 2: 'All Under Heaven' - Stage-based conquest, ranking by speed")
    print("\nQueries tested:")
    print("  1. LONG: Full 'Weathered Brilliance' description (stage-based, ranking by speed)")
    print("  2. SHORT: Realistic user search queries")
    print("\nExpected: Event 2 should have HIGHER similarity (same mechanics as query)")

    results = []

    if args.compare_both:
        # Test both providers
        from services.embedding_provider_factory import EmbeddingProviderFactory

        # Test Voyage
        try:
            voyage_provider = EmbeddingProviderFactory.create_provider(
                provider_type="voyage",
                model="voyage-3-large"
            )
            results.append(test_provider("Voyage", voyage_provider))
        except Exception as e:
            print(f"\n❌ Voyage test failed: {e}")

        # Test OpenAI
        try:
            openai_provider = EmbeddingProviderFactory.create_provider(
                provider_type="openai",
                model="text-embedding-3-large"
            )
            results.append(test_provider("OpenAI", openai_provider))
        except Exception as e:
            print(f"\n❌ OpenAI test failed: {e}")

    else:
        # Test single provider from env or argument
        provider_type = args.provider or os.getenv("EMBEDDING_PROVIDER", "voyage")

        from services.embedding_provider_factory import EmbeddingProviderFactory

        try:
            provider = EmbeddingProviderFactory.create_provider(
                provider_type=provider_type
            )
            results.append(test_provider(provider_type, provider))
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"\n{'Provider':<15} {'Model':<30} {'Correct Rate':<15}")
        print("-" * 60)
        for r in results:
            rate_pct = f"{r['correct_rate']*100:.0f}%"
            print(f"{r['provider']:<15} {r['model']:<30} {rate_pct:<15}")

        # Conclusion
        print("\n" + "-" * 60)
        avg_rate = sum(r['correct_rate'] for r in results) / len(results)
        if avg_rate >= 0.6:
            print(f"✅ Average correct rate: {avg_rate*100:.0f}% - Embeddings are working reasonably well")
        else:
            print(f"⚠️ Average correct rate: {avg_rate*100:.0f}% - May need to rely more on LLM reranking")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
