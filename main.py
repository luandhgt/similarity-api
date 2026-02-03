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

from fastapi import FastAPI, HTTPException
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

        # Initialize ServiceFactory to populate container with all services
        from core.service_factory import ServiceFactory
        from core.container import get_container, ServiceNames

        print("📦 Initializing services via ServiceFactory...")
        await ServiceFactory.initialize_all(verbose=True)
        container = get_container()
        print(f"✅ ServiceFactory initialized: {len(container.list_services())} services registered")

        # Load image model for similarity
        from models.places365 import get_places365_model
        get_places365_model()
        print("✅ Places365 model loaded")

        # Initialize Embedding provider for similarity (Voyage, OpenAI, Cohere based on EMBEDDING_PROVIDER env)
        from utils.text_processor import get_embedding_provider
        embedding_provider = get_embedding_provider()
        provider_info = embedding_provider.get_provider_info()
        print(f"✅ Embedding provider initialized: {provider_info['provider']} ({provider_info['model']}, {provider_info['dimensions']}d)")

        # Test LLM provider (should already be initialized by ServiceFactory)
        try:
            llm_provider = container.get(ServiceNames.CLAUDE)
            if llm_provider:
                provider_info = llm_provider.get_provider_info()
                print(f"✅ LLM provider ready: {provider_info['provider']} ({provider_info['model']})")
            else:
                print(f"⚠️ LLM provider: Not initialized")
        except Exception as e:
            print(f"⚠️ LLM provider warning: {e}")
        
        # Test configuration loading for about extraction
        try:
            from utils.prompt_manager import PromptManager
            from utils.output_formatter import output_formatter

            prompt_manager = PromptManager()
            available_prompts = prompt_manager.get_available_prompt_categories()
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
            from services.event_similarity_service import initialize_event_similarity_service
            from core.container import get_container, ServiceNames
            from utils.text_processor import get_embedding_provider

            # Get service dependencies using ServiceContainer (multi-provider support)
            container = get_container()
            llm_provider = container.get(ServiceNames.CLAUDE)  # Returns ChatGPT or Claude based on config
            embedding_provider = get_embedding_provider()
            database_service = app.state.database_service

            # Initialize service with all dependencies (text-only mode)
            event_similarity_service = initialize_event_similarity_service(
                claude_service=llm_provider,  # Actually LLM provider (Claude/ChatGPT/Gemini)
                embedding_provider=embedding_provider,
                database_service=database_service
            )

            # Store service instance in app state
            app.state.event_similarity_service = event_similarity_service

            # Test service status
            service_status = await event_similarity_service.get_service_status()
            print(f"✅ Event Similarity service ready: {service_status['status']} (text-only)")
        except Exception as e:
            print(f"⚠️ Event Similarity service warning: {e}")
            app.state.event_similarity_service = None

        # Initialize Determine Alternative service
        try:
            from services.determine_alternative_service import initialize_determine_alternative_service
            from core.container import get_container, ServiceNames

            container = get_container()
            llm_provider = container.get(ServiceNames.CLAUDE)
            database_service = app.state.database_service

            determine_alternative_service = initialize_determine_alternative_service(
                llm_provider=llm_provider,
                database_service=database_service
            )

            app.state.determine_alternative_service = determine_alternative_service
            print("✅ Determine Alternative service ready")
        except Exception as e:
            print(f"⚠️ Determine Alternative service warning: {e}")
            app.state.determine_alternative_service = None
        
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


# Dependency override for determine alternative service
async def get_determine_alternative_service():
    """Dependency to provide DetermineAlternativeService instance"""
    if not hasattr(app.state, 'determine_alternative_service') or app.state.determine_alternative_service is None:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DetermineAlternativeService not available. Please check service configuration and try again."
        )
    return app.state.determine_alternative_service

# Import and include routers
from routers.similarity import router as similarity_router
from routers.about_extraction import router as about_extraction_router
from routers.event_similarity import router as event_similarity_router
from routers.event_similarity import get_event_similarity_service as get_event_similarity_service_dep
from routers.determine_alternative import router as determine_alternative_router
from routers.determine_alternative import get_determine_alternative_service as get_determine_alternative_service_dep

# Override the dependency in event similarity router
app.dependency_overrides[get_event_similarity_service_dep] = get_event_similarity_service

# Override the dependency in determine alternative router
app.dependency_overrides[get_determine_alternative_service_dep] = get_determine_alternative_service

# Include all routers
app.include_router(similarity_router)
app.include_router(about_extraction_router)
app.include_router(event_similarity_router)
app.include_router(determine_alternative_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information"""
    try:
        # Check if models are loadable
        from models.places365 import get_places365_model
        places_model = get_places365_model()
        
        from utils.text_processor import get_embedding_provider
        embedding_provider = get_embedding_provider()
        
        # Check LLM provider (Claude/ChatGPT/Gemini)
        llm_available = False
        try:
            from core.container import get_container, ServiceNames
            container = get_container()
            llm_provider = container.get(ServiceNames.CLAUDE)
            if llm_provider:
                provider_info = llm_provider.get_provider_info()
                llm_available = provider_info.get('status') == 'ready'
        except:
            pass
        
        # Check Database service
        database_available = hasattr(app.state, 'database_service') and app.state.database_service is not None
        
        # Check Event Similarity service
        event_similarity_available = hasattr(app.state, 'event_similarity_service') and app.state.event_similarity_service is not None
        
        # Check Determine Alternative service
        determine_alternative_available = hasattr(app.state, 'determine_alternative_service') and app.state.determine_alternative_service is not None

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
                    "description": "Extract and synthesize about content from images using LLM (Claude/ChatGPT/Gemini)",
                    "endpoints": ["/api/extract-about", "/api/extract-about/status", "/api/extract-about/formats"]
                },
                "event_similarity": {
                    "description": "Comprehensive event similarity analysis with multi-modal search",
                    "endpoints": ["/api/find-similar-events", "/api/find-similar-events/status", "/api/find-similar-events/taxonomy"]
                },
                "determine_alternative": {
                    "description": "Determine if a new event is an alternative of existing candidate events",
                    "endpoints": ["/api/determine-alternative"]
                }
            },
            "models_loaded": {
                "places365": places_model is not None,
                "embedding_provider": embedding_provider is not None,
                "llm_provider": llm_available,
                "database": database_available,
                "event_similarity": event_similarity_available,
                "determine_alternative": determine_alternative_available
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
                "embedding_provider": False,
                "llm_provider": False,
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
        
        from utils.text_processor import get_embedding_provider
        embedding_provider = get_embedding_provider()
        embedding_info = embedding_provider.get_provider_info() if embedding_provider else {}
        health_status["services"]["embedding_provider"] = {
            "available": embedding_provider is not None,
            "provider": embedding_info.get('provider'),
            "model": embedding_info.get('model'),
            "dimensions": embedding_info.get('dimensions')
        }
        
        # Check LLM provider (Claude/ChatGPT/Gemini)
        try:
            from core.container import get_container, ServiceNames
            container = get_container()
            llm_provider = container.get(ServiceNames.CLAUDE)
            if llm_provider:
                provider_info = llm_provider.get_provider_info()
                health_status["services"]["llm_provider"] = {
                    "available": provider_info.get('status') == 'ready',
                    "provider": provider_info.get('provider'),
                    "model": provider_info.get('model')
                }
            else:
                health_status["services"]["llm_provider"] = {"available": False}
        except:
            health_status["services"]["llm_provider"] = {"available": False}
        
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

        # Check Determine Alternative service
        health_status["services"]["determine_alternative"] = (
            hasattr(app.state, 'determine_alternative_service') and
            app.state.determine_alternative_service is not None
        )

        # Overall health based on critical services
        critical_services = ["places365"]
        embedding_ok = health_status["services"].get("embedding_provider", {}).get("available", False)
        if all(health_status["services"].get(service, False) for service in critical_services) and embedding_ok:
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


# Global exception handlers for this router
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with proper error response format"""
    return {
        "error": "HTTPException",
        "message": exc.detail,
        "status_code": exc.status_code
    }

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors"""
    return {
        "error": "ValidationError",
        "message": str(exc),
        "status_code": 400
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