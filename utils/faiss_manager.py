#!/usr/bin/env python3
"""
FAISS Index Manager for Game Event Taxonomy
Supports multiple games and content types (about, images, name)
Structure: index/{content_type}/{game_code}_index.bin
"""
import logging
import faiss
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Union, Dict, Any

logger = logging.getLogger(__name__)

# Global FAISS indexes cache
_indexes_cache = {}

# Index structure
INDEX_ROOT = "index"
CONTENT_TYPES = ["about", "images", "name"]
VECTOR_DIMENSIONS = {
    "about": 1024,   # Text: Voyage-3-Large
    "images": 2048,  # Images: Places365
    "name": 1024     # Text: Voyage-3-Large
}


def normalize_game_code(game_name: str) -> str:
    """
    Convert game name to normalized code
    Example: "Best Fiends" -> "best_fiends"
    """
    return game_name.lower().replace(" ", "_").replace("-", "_")


def get_index_path(game_code: str, content_type: str) -> Path:
    """
    Get path for specific game and content type index
    
    Args:
        game_code: Normalized game code (e.g., "best_fiends")
        content_type: "about", "images", or "name"
        
    Returns:
        Path: Path to index file
    """
    if content_type not in CONTENT_TYPES:
        raise ValueError(f"content_type must be one of {CONTENT_TYPES}, got '{content_type}'")
    
    index_dir = Path(INDEX_ROOT) / content_type
    index_dir.mkdir(parents=True, exist_ok=True)
    
    return index_dir / f"{game_code}_index.bin"


def get_index_key(game_code: str, content_type: str) -> str:
    """Generate cache key for index"""
    return f"{game_code}_{content_type}"


def load_faiss_index(game_code: str, content_type: str):
    """
    Load specific FAISS index for game and content type
    
    Args:
        game_code: Normalized game code
        content_type: "about", "images", or "name"
        
    Returns:
        faiss.Index: FAISS index
    """
    global _indexes_cache
    
    # Check cache first
    cache_key = get_index_key(game_code, content_type)
    if cache_key in _indexes_cache:
        return _indexes_cache[cache_key]
    
    # Get paths and dimensions
    index_path = get_index_path(game_code, content_type)
    dimension = VECTOR_DIMENSIONS[content_type]
    
    # Load or create index
    if index_path.exists():
        print(f"📂 Loading {game_code}/{content_type} index: {index_path}")
        try:
            index = faiss.read_index(str(index_path))
            print(f"✅ Loaded {game_code}/{content_type} index with {index.ntotal} vectors")
        except Exception as e:
            print(f"❌ Failed to load {game_code}/{content_type} index: {e}")
            print(f"🔧 Creating new {game_code}/{content_type} index...")
            index = faiss.IndexFlatIP(dimension)
    else:
        print(f"🔧 Creating new {game_code}/{content_type} index ({dimension}d)")
        index = faiss.IndexFlatIP(dimension)
    
    # Cache the index
    _indexes_cache[cache_key] = index
    
    return index


def save_faiss_index(game_code: str, content_type: str):
    """
    Save specific FAISS index
    
    Args:
        game_code: Normalized game code
        content_type: "about", "images", or "name"
    """
    global _indexes_cache
    
    cache_key = get_index_key(game_code, content_type)
    
    if cache_key not in _indexes_cache:
        print(f"⚠️ No index loaded for {game_code}/{content_type}")
        return
    
    index = _indexes_cache[cache_key]
    index_path = get_index_path(game_code, content_type)
    
    try:
        faiss.write_index(index, str(index_path))
        print(f"💾 Saved {game_code}/{content_type} index with {index.ntotal} vectors")
    except Exception as e:
        print(f"❌ Failed to save {game_code}/{content_type} index: {e}")


def add_vector_to_faiss(vector: Union[np.ndarray, list], 
                       game_name: str, 
                       content_type: str) -> int:
    """
    Add vector to appropriate FAISS index
    
    Args:
        vector: numpy array or list
        game_name: Game name (will be normalized to code)
        content_type: "about", "images", or "name"
        
    Returns:
        int: faiss_index (position in FAISS array)
    """
    
    # Normalize game name to code
    game_code = normalize_game_code(game_name)
    
    # Validate content_type
    if content_type not in CONTENT_TYPES:
        raise ValueError(f"content_type must be one of {CONTENT_TYPES}, got '{content_type}'")
    
    # Load index
    index = load_faiss_index(game_code, content_type)
    
    # Process vector
    if isinstance(vector, list):
        vector = np.array(vector, dtype=np.float32)
    
    vector = vector.astype(np.float32)
    expected_dim = VECTOR_DIMENSIONS[content_type]
    
    if vector.shape != (expected_dim,):
        raise ValueError(f"Vector for {content_type} must be shape ({expected_dim},), got {vector.shape}")
    
    # Get current index position
    faiss_index = index.ntotal
    
    # Reshape for FAISS (needs 2D array)
    vector_2d = vector.reshape(1, -1)
    
    # Add to index
    index.add(vector_2d)
    
    print(f"✅ Added vector to {game_code}/{content_type} at index {faiss_index}, total: {index.ntotal}")
    
    # Auto-save
    save_faiss_index(game_code, content_type)
    
    return faiss_index


def search_similar_vectors(vector: Union[np.ndarray, list],
                         content_type: str,
                         game_code: str,
                         top_k: int = 10) -> list:
    """
    Search for similar vectors in specific game/content index
    
    Args:
        vector: Query vector
        content_type: "about", "images", or "name"
        game_code: Normalized game code  
        top_k: Number of results to return
        
    Returns:
        list: List of dicts with 'index' and 'score' keys
    """
    
    index = load_faiss_index(game_code, content_type)
    
    if index.ntotal == 0:
        return []
    
    # Process query vector
    if isinstance(vector, list):
        vector = np.array(vector, dtype=np.float32)
    
    vector = vector.astype(np.float32).reshape(1, -1)
    
    # Search
    distances, indices = index.search(vector, min(top_k, index.ntotal))
    
    # Format results as expected by event_similarity_service
    results = []
    for i, (score, idx) in enumerate(zip(distances[0], indices[0])):
        results.append({
            'index': int(idx),
            'score': float(score)
        })
    
    return results

def get_vector_by_index(faiss_index: int, content_type: str, game_code: str) -> np.ndarray:
    """
    Retrieve a specific vector from FAISS index by its index number
    
    Args:
        faiss_index: FAISS index position
        content_type: "about", "images", or "name"  
        game_code: Normalized game code
        
    Returns:
        np.ndarray: Vector at the specified index, or None if not found
    """
    try:
        index = load_faiss_index(game_code, content_type)
        
        if index.ntotal == 0:
            print(f"Index {game_code}/{content_type} is empty")
            return None
            
        if faiss_index >= index.ntotal or faiss_index < 0:
            print(f"FAISS index {faiss_index} out of range [0, {index.ntotal})")
            return None
        
        # Get the vector at specified index
        vector = index.reconstruct(faiss_index)
        
        return vector.astype(np.float32)
        
    except Exception as e:
        print(f"Error retrieving vector at index {faiss_index}: {e}")
        return None


def get_faiss_stats(game_name: str = None, content_type: str = None) -> Dict[str, Any]:
    """
    Get FAISS index statistics
    
    Args:
        game_name: Optional specific game (if None, returns all games)
        content_type: Optional specific content type (if None, returns all types)
        
    Returns:
        dict: Statistics
    """
    
    if game_name and content_type:
        # Single index stats
        game_code = normalize_game_code(game_name)
        index = load_faiss_index(game_code, content_type)
        index_path = get_index_path(game_code, content_type)
        
        index_size_mb = 0
        if index_path.exists():
            index_size_mb = index_path.stat().st_size / 1024 / 1024
        
        return {
            "game": game_name,
            "game_code": game_code,
            "content_type": content_type,
            "total_vectors": index.ntotal,
            "vector_dimension": VECTOR_DIMENSIONS[content_type],
            "index_file": str(index_path),
            "index_size_mb": round(index_size_mb, 2)
        }
    
    else:
        # All indexes stats
        stats = {
            "structure": {
                "index_root": INDEX_ROOT,
                "content_types": CONTENT_TYPES,
                "vector_dimensions": VECTOR_DIMENSIONS
            },
            "games": {}
        }
        
        # Scan existing index files
        index_root = Path(INDEX_ROOT)
        if index_root.exists():
            for content_dir in index_root.iterdir():
                if content_dir.is_dir() and content_dir.name in CONTENT_TYPES:
                    content_type = content_dir.name
                    
                    for index_file in content_dir.glob("*_index.bin"):
                        game_code = index_file.stem.replace("_index", "")
                        
                        if game_code not in stats["games"]:
                            stats["games"][game_code] = {}
                        
                        # Get index info
                        try:
                            index = load_faiss_index(game_code, content_type)
                            index_size_mb = index_file.stat().st_size / 1024 / 1024
                            
                            stats["games"][game_code][content_type] = {
                                "total_vectors": index.ntotal,
                                "vector_dimension": VECTOR_DIMENSIONS[content_type],
                                "index_size_mb": round(index_size_mb, 2)
                            }
                        except Exception as e:
                            stats["games"][game_code][content_type] = {"error": str(e)}
        
        return stats


def cleanup_faiss():
    """Cleanup and save all loaded indexes"""
    global _indexes_cache
    
    print("🧹 Cleaning up FAISS indexes...")
    
    for cache_key, index in _indexes_cache.items():
        try:
            game_code, content_type = cache_key.rsplit("_", 1)
            save_faiss_index(game_code, content_type)
        except Exception as e:
            print(f"❌ Failed to save {cache_key}: {e}")
    
    print("✅ FAISS cleanup completed")


def list_available_games() -> Dict[str, list]:
    """
    List all available games and their content types
    
    Returns:
        dict: {game_code: [content_types]}
    """
    games = {}
    index_root = Path(INDEX_ROOT)
    
    if index_root.exists():
        for content_dir in index_root.iterdir():
            if content_dir.is_dir() and content_dir.name in CONTENT_TYPES:
                content_type = content_dir.name
                
                for index_file in content_dir.glob("*_index.bin"):
                    game_code = index_file.stem.replace("_index", "")
                    
                    if game_code not in games:
                        games[game_code] = []
                    
                    if content_type not in games[game_code]:
                        games[game_code].append(content_type)
    
    return games