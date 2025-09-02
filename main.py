#!/usr/bin/env python3
"""
AI Service - Main FastAPI application
Supports:
- Image and text embedding with similarity search (existing)
- About content extraction from images using Claude AI (existing)
- Event similarity analysis with multi-modal search (new)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time
from contextlib import asynccontextmanager
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    try:
        print("🚀 Starting AI Service...")
        
        # Load image model for similarity
        from models.places365 import get_places365_model
        get_places365_model()
        print("✅ Places365 model loaded")
        
        # Initialize Voyage client for similarity
        from utils.text_processor import get_voyage_client
        get_voyage_client()
        print("✅ Voyage client initialized")
        
        # Test Claude service for about extraction
        try:
            from services.claude_service import claude_service
            status = claude_service.get_usage_stats()
            print(f"✅ Claude service ready: {status['model']}")
        except Exception as e:
            print(f"⚠️ Claude service warning: {e}")
        
        # Test configuration loading for about extraction
        try:
            from utils.prompt_manager import prompt_manager
            from utils.output_formatter import output_formatter
            
            available_prompts = prompt_manager.get_available_categories()
            available_formats = output_formatter.get_available_formats()
            
            print(f"✅ Loaded {len(available_prompts)} prompt categories: {available_prompts}")
            print(f"✅ Loaded {len(available_formats)} output formats: {available_formats}")
        except Exception as e:
            print(f"⚠️ About extraction config warning: {e}")
        
        # Initialize Database service for event similarity
        try:
            from services.database_service import DatabaseService
            db_service = DatabaseService()
            await db_service.initialize()
            
            # Store database service instance in app state
            app.state.database_service = db_service
            print("✅ Database service initialized")
        except Exception as e:
            print(f"⚠️ Database service warning: {e}")
            app.state.database_service = None
        
        # Initialize Event Similarity service
        try:
            from services.event_similarity_service import EventSimilarityService
            
            # Create service instance with dependencies
            event_similarity_service = EventSimilarityService(
                database_service=app.state.database_service
            )
            
            # Store service instance in app state
            app.state.event_similarity_service = event_similarity_service
            
            # Test service status
            service_status = await event_similarity_service.get_service_status()
            print(f"✅ Event Similarity service ready: {service_status['status']}")
        except Exception as e:
            print(f"⚠️ Event Similarity service warning: {e}")
            app.state.event_similarity_service = None
        
        print("✅ API ready")
    except Exception as e:
        print(f"❌ Startup failed: {e}")
    
    yield  # API running
    
    # Shutdown logic
    print("🔥 Shutting down...")
    try:
        from utils.faiss_manager import cleanup_faiss
        cleanup_faiss()
        print("✅ FAISS cleanup complete")
    except:
        pass
    
    # Cleanup database connections
    try:
        if hasattr(app.state, 'database_service') and app.state.database_service:
            await app.state.database_service.close()
            print("✅ Database connections closed")
    except Exception as e:
        print(f"⚠️ Database cleanup warning: {e}")
    
    print("👋 Shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title="AI Service - Similarity, About Extraction & Event Analysis",
    description="Multi-game image/text similarity search + AI-powered about content extraction + comprehensive event similarity analysis",
    version="3.0.0",
    lifespan=lifespan  
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency override for event similarity service
async def get_event_similarity_service():
    """Dependency to provide EventSimilarityService instance"""
    if not hasattr(app.state, 'event_similarity_service') or app.state.event_similarity_service is None:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="EventSimilarityService not available. Please check service configuration and try again."
        )
    return app.state.event_similarity_service

# Import and include routers
from routers.similarity import router as similarity_router
from routers.about_extraction import router as about_extraction_router
from routers.event_similarity import router as event_similarity_router
from routers.event_similarity import get_event_similarity_service as get_event_similarity_service_dep

# Override the dependency in event similarity router
app.dependency_overrides[get_event_similarity_service_dep] = get_event_similarity_service

# Include all routers
app.include_router(similarity_router)
app.include_router(about_extraction_router)
app.include_router(event_similarity_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information"""
    try:
        # Check if models are loadable
        from models.places365 import get_places365_model
        places_model = get_places365_model()
        
        from utils.text_processor import get_voyage_client
        voyage_client = get_voyage_client()
        
        # Check Claude service
        claude_available = False
        try:
            from services.claude_service import claude_service
            claude_status = claude_service.get_usage_stats()
            claude_available = True
        except:
            pass
        
        # Check Database service
        database_available = hasattr(app.state, 'database_service') and app.state.database_service is not None
        
        # Check Event Similarity service
        event_similarity_available = hasattr(app.state, 'event_similarity_service') and app.state.event_similarity_service is not None
        
        return {
            "service": "AI Service - Similarity, About Extraction & Event Analysis",
            "version": "3.0.0",
            "status": "ready",
            "services": {
                "similarity_search": {
                    "description": "Image and text embedding with similarity search",
                    "endpoints": ["/embed_image", "/embed_text", "/search", "/stats", "/games"]
                },
                "about_extraction": {
                    "description": "Extract and synthesize about content from images using Claude AI",
                    "endpoints": ["/api/extract-about", "/api/extract-about/status", "/api/extract-about/formats"]
                },
                "event_similarity": {
                    "description": "Comprehensive event similarity analysis with multi-modal search",
                    "endpoints": ["/api/find-similar-events", "/api/find-similar-events/status", "/api/find-similar-events/taxonomy"]
                }
            },
            "models_loaded": {
                "places365": places_model is not None,
                "voyage": voyage_client is not None,
                "claude": claude_available,
                "database": database_available,
                "event_similarity": event_similarity_available
            },
            "documentation": "/docs"
        }
    except Exception as e:
        return {
            "service": "AI Service",
            "status": "error",
            "error": str(e),
            "models_loaded": {
                "places365": False, 
                "voyage": False, 
                "claude": False,
                "database": False,
                "event_similarity": False
            }
        }

# Global health check
@app.get("/health")
async def health_check():
    """Enhanced health check including all services"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {}
    }
    
    try:
        # Check basic models
        from models.places365 import get_places365_model
        places_model = get_places365_model()
        health_status["services"]["places365"] = places_model is not None
        
        from utils.text_processor import get_voyage_client  
        voyage_client = get_voyage_client()
        health_status["services"]["voyage"] = voyage_client is not None
        
        # Check Claude service
        try:
            from services.claude_service import claude_service
            claude_status = claude_service.get_usage_stats()
            health_status["services"]["claude"] = True
        except:
            health_status["services"]["claude"] = False
        
        # Check Database service
        if hasattr(app.state, 'database_service') and app.state.database_service:
            try:
                db_health = await app.state.database_service.health_check()
                health_status["services"]["database"] = db_health.get("status") == "healthy"
            except:
                health_status["services"]["database"] = False
        else:
            health_status["services"]["database"] = False
        
        # Check Event Similarity service
        if hasattr(app.state, 'event_similarity_service') and app.state.event_similarity_service:
            try:
                event_status = await app.state.event_similarity_service.get_service_status()
                health_status["services"]["event_similarity"] = event_status.get("status") == "healthy"
            except:
                health_status["services"]["event_similarity"] = False
        else:
            health_status["services"]["event_similarity"] = False
        
        # Overall health based on critical services
        critical_services = ["places365", "voyage"]
        if all(health_status["services"].get(service, False) for service in critical_services):
            health_status["status"] = "healthy"
        else:
            health_status["status"] = "degraded"
            
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
    
    return health_status

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"❌ Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "status_code": 500
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Service server...")
    uvicorn.run(
        "main:app",  # Use import string instead of app object
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        reload=True
    )