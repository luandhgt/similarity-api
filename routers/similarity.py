"""
Similarity Router - Image and text embedding with similarity search
Extracted from original main.py
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import time
from typing import Optional, List

# Import processors
from utils.image_processor import extract_image_features, validate_image_file
from utils.text_processor import extract_text_features, validate_text_input, get_voyage_client
from utils.faiss_manager import (
    add_vector_to_faiss, 
    search_similar_vectors,
    get_faiss_stats, 
    list_available_games,
    normalize_game_code,
    CONTENT_TYPES
)

router = APIRouter(tags=["similarity"])

# Pydantic models
class EmbedImageRequest(BaseModel):
    image_path: str
    game_name: str
    content_type: str = "images"  # Default to images

class EmbedTextRequest(BaseModel):
    text: str
    game_name: str
    content_type: str  # "about" or "name"

class SearchRequest(BaseModel):
    game_name: str
    content_type: str
    top_k: int = 10
    # Either image_path OR text (not both)
    image_path: Optional[str] = None
    text: Optional[str] = None

class EmbedResponse(BaseModel):
    success: bool
    faiss_index: int
    processing_time: float
    vector_dimension: int
    game_code: str
    content_type: str

class SearchResult(BaseModel):
    faiss_index: int
    distance: float

class SearchResponse(BaseModel):
    success: bool
    results: List[SearchResult]
    processing_time: float
    game_code: str
    content_type: str
    total_found: int

class HealthResponse(BaseModel):
    message: str
    status: str
    models_loaded: dict

# API Endpoints
@router.post("/embed_image", response_model=EmbedResponse)
async def embed_image(request: EmbedImageRequest):
    """
    Extract feature vector from image file using Places365
    
    Args:
        request: Contains image_path, game_name, and content_type
        
    Returns:
        EmbedResponse with faiss_index and processing details
    """
    start_time = time.time()
    
    try:
        # Validate content_type for images
        if request.content_type != "images":
            raise HTTPException(
                status_code=400,
                detail=f"content_type for images must be 'images', got '{request.content_type}'"
            )
        
        # Validate input
        validate_image_file(request.image_path)
        
        # Extract features
        vector = extract_image_features(request.image_path)
        
        # Add to FAISS
        faiss_index = add_vector_to_faiss(vector, request.game_name, request.content_type)
        
        processing_time = time.time() - start_time
        
        return EmbedResponse(
            success=True,
            faiss_index=faiss_index,
            processing_time=round(processing_time, 3),
            vector_dimension=len(vector),
            game_code=normalize_game_code(request.game_name),
            content_type=request.content_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process image: {str(e)}"
        )

@router.post("/embed_text", response_model=EmbedResponse)
async def embed_text(request: EmbedTextRequest):
    """
    Extract feature vector from text using Voyage-3-Large
    
    Args:
        request: Contains text, game_name, and content_type
        
    Returns:
        EmbedResponse with faiss_index and processing details
    """
    start_time = time.time()
    
    try:
        # Validate content_type for text
        if request.content_type not in ["about", "name"]:
            raise HTTPException(
                status_code=400,
                detail=f"content_type for text must be 'about' or 'name', got '{request.content_type}'"
            )
        
        # Validate input
        validate_text_input(request.text)
        
        # Extract features
        vector = extract_text_features(request.text)
        
        # Add to FAISS
        faiss_index = add_vector_to_faiss(vector, request.game_name, request.content_type)
        
        processing_time = time.time() - start_time
        
        return EmbedResponse(
            success=True,
            faiss_index=faiss_index,
            processing_time=round(processing_time, 3),
            vector_dimension=len(vector),
            game_code=normalize_game_code(request.game_name),
            content_type=request.content_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process text: {str(e)}"
        )

@router.post("/search", response_model=SearchResponse)
async def search_similar(request: SearchRequest):
    """
    Search for similar vectors in specific game/content index
    
    Args:
        request: Contains query (image_path OR text), game_name, content_type, top_k
        
    Returns:
        SearchResponse with similar vectors and distances
    """
    start_time = time.time()
    
    try:
        # Validate that exactly one of image_path or text is provided
        if bool(request.image_path) == bool(request.text):
            raise HTTPException(
                status_code=400,
                detail="Must provide exactly one of 'image_path' or 'text', not both or neither"
            )
        
        # Validate content_type
        if request.content_type not in CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"content_type must be one of {CONTENT_TYPES}, got '{request.content_type}'"
            )
        
        # Extract query vector
        if request.image_path:
            # Image search
            if request.content_type != "images":
                raise HTTPException(
                    status_code=400,
                    detail="image_path can only be used with content_type='images'"
                )
            
            validate_image_file(request.image_path)
            query_vector = extract_image_features(request.image_path)
            
        else:
            # Text search
            if request.content_type not in ["about", "name"]:
                raise HTTPException(
                    status_code=400,
                    detail="text can only be used with content_type='about' or 'name'"
                )
            
            validate_text_input(request.text)
            query_vector = extract_text_features(request.text)
        
        # Search
        distances, indices = search_similar_vectors(
            query_vector, 
            request.game_name, 
            request.content_type, 
            request.top_k
        )
        
        # Format results
        results = [
            SearchResult(faiss_index=int(idx), distance=float(dist))
            for idx, dist in zip(indices, distances)
        ]
        
        processing_time = time.time() - start_time
        
        return SearchResponse(
            success=True,
            results=results,
            processing_time=round(processing_time, 3),
            game_code=normalize_game_code(request.game_name),
            content_type=request.content_type,
            total_found=len(results)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search: {str(e)}"
        )

@router.get("/stats")
async def get_all_stats():
    """Get all FAISS index statistics"""
    try:
        stats = get_faiss_stats()
        return {
            "faiss_stats": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": time.time()
        }

@router.get("/stats/{game_name}/{content_type}")
async def get_game_stats(game_name: str, content_type: str):
    """Get specific game/content statistics"""
    try:
        if content_type not in CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"content_type must be one of {CONTENT_TYPES}"
            )
        
        stats = get_faiss_stats(game_name, content_type)
        return {
            "faiss_stats": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": time.time()
        }

@router.get("/games")
async def list_games():
    """List all available games and their content types"""
    try:
        games = list_available_games()
        return {
            "games": games,
            "total_games": len(games),
            "content_types": CONTENT_TYPES,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": time.time()
        }

@router.get("/model_info")
async def model_info():
    """Get model information"""
    try:
        from models.places365 import get_places365_model
        places_model = get_places365_model()
        
        # Count parameters
        total_params = sum(p.numel() for p in places_model.parameters())
        trainable_params = sum(p.numel() for p in places_model.parameters() if p.requires_grad)
        
        return {
            "models": {
                "places365": {
                    "name": "Places365 ResNet50",
                    "vector_dimension": 2048,
                    "content_type": "images",
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "loaded": True
                },
                "voyage": {
                    "name": "Voyage-3-Large",
                    "vector_dimension": 1024,
                    "content_types": ["about", "name"],
                    "loaded": get_voyage_client() is not None
                }
            },
            "supported_content_types": CONTENT_TYPES
        }
    except Exception as e:
        return {
            "error": str(e),
            "models": {"places365": {"loaded": False}, "voyage": {"loaded": False}}
        }