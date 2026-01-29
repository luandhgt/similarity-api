"""
Debug script to compare FAISS similarity between query event and specific events.

Usage:
    python debug_faiss_similarity.py
"""

import asyncio
import numpy as np
from pathlib import Path
import sys
import os
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
env_file = Path(__file__).parent / ".env.development"
if env_file.exists():
    load_dotenv(env_file)
    print(f"Loaded env from {env_file}")
else:
    load_dotenv()  # Try default .env
    print("Loaded env from .env")

from utils.text_processor import extract_text_features, VoyageClient
from utils.faiss_manager import get_vector_by_index, normalize_game_code, load_faiss_index


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def vectors_are_equal(vec1: np.ndarray, vec2: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Check if two vectors are equal within tolerance"""
    return np.allclose(vec1, vec2, atol=tolerance)


async def main():
    print("=" * 80)
    print("FAISS SIMILARITY DEBUG SCRIPT")
    print("=" * 80)

    # Initialize Voyage client
    print("\n1. Initializing Voyage client...")
    voyage_client = VoyageClient(api_key=os.getenv('VOYAGE_API_KEY'))
    print("   ✅ Voyage client ready")

    # Game config
    game_code = "Rise of Kingdoms"
    normalized_game_code = normalize_game_code(game_code)
    print(f"\n2. Game: {game_code} (normalized: {normalized_game_code})")

    # Query event: Weathered Brilliance
    query_about = '''The “Weathered Brilliance” event features a **sequential stage-based progression system** where players advance through multiple battle stages on a map. Each stage is represented by a banner, and progression follows a linear path. Players can replay previously cleared stages at any time, and completing all stages unlocks a **Hard Mode**, offering an additional layer of challenge for advanced players.

Ranking in the event is determined by **speed-based competition**. Players are ranked within their kingdom based on how quickly they clear each stage. The leaderboard distinguishes between those who complete stages with items and those without, emphasizing both efficiency and resource management. Rankings are visible after each stage is cleared, reinforcing the competitive aspect.

Rewards are distributed **per stage** and likely include in-game items such as scrolls and speedups, as shown in the event interface. Additional ranking rewards may be granted based on leaderboard performance, providing incentive to optimize both speed and strategy.

This event is **individualistic and competitive** in nature, as participation and performance are tracked per player rather than in groups. There is no visible evidence of cooperative mechanics.

Visually, the event carries a **military strategy theme**, with parchment maps, battlefield imagery, and a general/tactician character design. The aesthetic supports the tactical warfare concept central to the gameplay.'''

    # Event 1: War of the Ruins (faiss_index = 211)
    event1_faiss_index = 211
    event1_name = "War of the Ruins"

    # Event 2: All Under Heaven (faiss_index = 297)
    event2_faiss_index = 297
    event2_name = "All Under Heaven"

    print(f"\n3. Target events:")
    print(f"   - Event 1: {event1_name} (faiss_index = {event1_faiss_index})")
    print(f"   - Event 2: {event2_name} (faiss_index = {event2_faiss_index})")

    # Step 1: Generate embedding for query
    print("\n" + "=" * 80)
    print("STEP 1: Generate query embedding")
    print("=" * 80)

    query_vector = extract_text_features(query_about, voyage_client)
    print(f"   Query vector shape: {query_vector.shape}")
    print(f"   Query vector norm: {np.linalg.norm(query_vector):.6f}")
    print(f"   Query vector first 5 values: {query_vector[:5]}")

    # Step 2: Get vectors from FAISS for both events
    print("\n" + "=" * 80)
    print("STEP 2: Get vectors from FAISS index")
    print("=" * 80)

    # Load FAISS index info
    faiss_index = load_faiss_index(normalized_game_code, "about")
    print(f"   FAISS index total vectors: {faiss_index.ntotal}")

    event1_vector_faiss = get_vector_by_index(event1_faiss_index, "about", normalized_game_code)
    event2_vector_faiss = get_vector_by_index(event2_faiss_index, "about", normalized_game_code)

    if event1_vector_faiss is not None:
        print(f"\n   Event 1 ({event1_name}) - FAISS index {event1_faiss_index}:")
        print(f"      Vector shape: {event1_vector_faiss.shape}")
        print(f"      Vector norm: {np.linalg.norm(event1_vector_faiss):.6f}")
        print(f"      First 5 values: {event1_vector_faiss[:5]}")
    else:
        print(f"   ❌ Event 1 vector not found in FAISS!")

    if event2_vector_faiss is not None:
        print(f"\n   Event 2 ({event2_name}) - FAISS index {event2_faiss_index}:")
        print(f"      Vector shape: {event2_vector_faiss.shape}")
        print(f"      Vector norm: {np.linalg.norm(event2_vector_faiss):.6f}")
        print(f"      First 5 values: {event2_vector_faiss[:5]}")
    else:
        print(f"   ❌ Event 2 vector not found in FAISS!")

    # Step 3: Calculate direct cosine similarity
    print("\n" + "=" * 80)
    print("STEP 3: Calculate DIRECT cosine similarity (Query vs Events)")
    print("=" * 80)

    if event1_vector_faiss is not None:
        sim1 = cosine_similarity(query_vector, event1_vector_faiss)
        print(f"\n   Query vs Event 1 ({event1_name}): {sim1:.6f}")

    if event2_vector_faiss is not None:
        sim2 = cosine_similarity(query_vector, event2_vector_faiss)
        print(f"   Query vs Event 2 ({event2_name}): {sim2:.6f}")

    if event1_vector_faiss is not None and event2_vector_faiss is not None:
        print(f"\n   🎯 RESULT: Event {'1' if sim1 > sim2 else '2'} has HIGHER similarity!")
        print(f"      Difference: {abs(sim1 - sim2):.6f}")

    # Step 4: Verify FAISS search results
    print("\n" + "=" * 80)
    print("STEP 4: Verify FAISS search - what does FAISS actually return?")
    print("=" * 80)

    # Normalize query vector for search
    query_vector_search = query_vector.astype(np.float32).reshape(1, -1)

    # Search top 20
    distances, indices = faiss_index.search(query_vector_search, 20)

    print(f"\n   FAISS Top 20 results for query:")
    print(f"   {'Rank':<6} {'Index':<10} {'Distance':<12} {'Is Event 1?':<12} {'Is Event 2?':<12}")
    print("   " + "-" * 60)

    event1_found = False
    event2_found = False
    event1_rank = -1
    event2_rank = -1

    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        is_event1 = "✅ YES" if idx == event1_faiss_index else ""
        is_event2 = "✅ YES" if idx == event2_faiss_index else ""

        if idx == event1_faiss_index:
            event1_found = True
            event1_rank = rank
        if idx == event2_faiss_index:
            event2_found = True
            event2_rank = rank

        print(f"   {rank:<6} {idx:<10} {dist:<12.6f} {is_event1:<12} {is_event2:<12}")

    print(f"\n   📊 Summary:")
    print(f"      Event 1 ({event1_name}): {'Found at rank ' + str(event1_rank) if event1_found else '❌ NOT in top 20'}")
    print(f"      Event 2 ({event2_name}): {'Found at rank ' + str(event2_rank) if event2_found else '❌ NOT in top 20'}")

    # Step 5: Search with larger K to find Event 2
    if not event2_found:
        print("\n" + "=" * 80)
        print("STEP 5: Extended search - find Event 2's actual rank")
        print("=" * 80)

        # Search all vectors
        distances_all, indices_all = faiss_index.search(query_vector_search, faiss_index.ntotal)

        for rank, (idx, dist) in enumerate(zip(indices_all[0], distances_all[0]), 1):
            if idx == event2_faiss_index:
                print(f"\n   🔍 Event 2 ({event2_name}) found at rank {rank} with distance {dist:.6f}")

                # Get the event at that rank
                event_at_rank_10 = indices_all[0][9]  # 0-indexed, so rank 10 is index 9
                dist_at_rank_10 = distances_all[0][9]
                print(f"   📌 For comparison, rank 10 has index {event_at_rank_10} with distance {dist_at_rank_10:.6f}")
                break
        else:
            print(f"\n   ❌ Event 2 not found in entire index!")

    # Step 6: Compare similarity between Event 1 and Event 2
    print("\n" + "=" * 80)
    print("STEP 6: Compare Event 1 vs Event 2 directly")
    print("=" * 80)

    if event1_vector_faiss is not None and event2_vector_faiss is not None:
        sim_1_vs_2 = cosine_similarity(event1_vector_faiss, event2_vector_faiss)
        print(f"\n   Event 1 vs Event 2 similarity: {sim_1_vs_2:.6f}")
        print(f"   (This shows how similar the two events are to each other)")

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
