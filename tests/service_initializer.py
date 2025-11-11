"""
Service Initializer Module

Handles initialization of all required services for testing.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


async def initialize_services(verbose: bool = True) -> Dict[str, Any]:
    """
    Initialize all required services for testing

    Args:
        verbose: Whether to print verbose output

    Returns:
        Dictionary containing service instances and status
    """
    if verbose:
        print("🔧 Initializing services...")

    services = {}

    try:
        # Initialize Places365 model
        if verbose:
            print("Loading Places365 model...")
        from models.places365 import get_places365_model
        places_model = get_places365_model()
        services['places365'] = places_model is not None
        if verbose:
            print(f"{'✅' if services['places365'] else '❌'} Places365 model: {'loaded' if services['places365'] else 'failed'}")

        # Initialize Voyage client
        if verbose:
            print("Initializing Voyage client...")
        from utils.text_processor import get_voyage_client
        voyage_client = get_voyage_client()
        services['voyage'] = voyage_client is not None
        services['voyage_client'] = voyage_client
        if verbose:
            print(f"{'✅' if services['voyage'] else '❌'} Voyage client: {'initialized' if services['voyage'] else 'failed'}")

        # Initialize Claude service
        if verbose:
            print("Testing Claude service...")
        try:
            from services.claude_service import ClaudeService
            claude_service = ClaudeService()
            claude_status = claude_service.get_usage_stats()
            services['claude'] = True
            services['claude_service'] = claude_service
            if verbose:
                print(f"✅ Claude service: ready ({claude_status.get('model', 'unknown model')})")
        except Exception as e:
            services['claude'] = False
            services['claude_service'] = None
            if verbose:
                print(f"⚠️ Claude service: {e}")
                print(f"   Make sure CLAUDE_API_KEY is set in .env file")

        # Initialize Database service
        if verbose:
            print("Initializing Database service...")
        try:
            from services.database_service import DatabaseService
            db_service = DatabaseService()
            await db_service.initialize()

            # Test database connection
            db_health = await db_service.health_check()
            services['database'] = db_health.get("status") == "healthy"
            services['database_service'] = db_service
            if verbose:
                print(f"{'✅' if services['database'] else '❌'} Database service: {'connected' if services['database'] else 'failed'}")
        except Exception as e:
            services['database'] = False
            services['database_service'] = None
            if verbose:
                print(f"⚠️ Database service: {e}")

        # Initialize Event Similarity service
        if verbose:
            print("Initializing Event Similarity service...")
        try:
            # Check if config files exist first
            config_path = Path("config")
            required_configs = [
                "prompts.yaml",
                "output_formats.yaml",
                "similarity_prompts.yaml",
                "similarity_output_formats.yaml"
            ]

            missing_configs = []
            for config_file in required_configs:
                if not (config_path / config_file).exists():
                    missing_configs.append(config_file)

            if missing_configs and verbose:
                print(f"⚠️ Missing config files: {missing_configs}")
                print(f"   Expected location: {config_path.absolute()}/")

            # Import required classes
            from services.event_similarity_service import EventSimilarityService
            from utils.prompt_manager import PromptManager

            # Check that all dependencies are available
            if not services.get('claude_service'):
                raise Exception("Claude service not available")
            if not services.get('voyage_client'):
                raise Exception("Voyage client not available")
            if not services.get('database_service'):
                raise Exception("Database service not available")

            # Initialize PromptManager
            prompt_manager = PromptManager()

            # Create service instance with ALL required dependencies
            event_similarity_service = EventSimilarityService(
                claude_service=services['claude_service'],
                voyage_client=services['voyage_client'],
                prompt_manager=prompt_manager,
                database_service=services['database_service']
            )

            # Test service status
            service_status = await event_similarity_service.get_service_status()
            services['event_similarity'] = service_status.get("status") == "operational"
            services['event_similarity_service'] = event_similarity_service
            if verbose:
                print(f"{'✅' if services['event_similarity'] else '❌'} Event Similarity service: {'ready' if services['event_similarity'] else 'failed'}")

            if verbose and services['event_similarity']:
                db_status = service_status.get('database_health', {})
                print(f"   Database connected: {db_status.get('status') == 'healthy'}")
                faiss_stats = service_status.get('faiss_stats', {})
                faiss_count = len([k for k, v in faiss_stats.items() if isinstance(v, dict) and v.get('total_vectors', 0) > 0])
                print(f"   FAISS indexes loaded: {faiss_count}")

        except ImportError as e:
            services['event_similarity'] = False
            services['event_similarity_service'] = None
            if verbose:
                print(f"❌ Event Similarity service - Import Error: {e}")
                if "sklearn" in str(e):
                    print(f"   Solution: pip install scikit-learn==1.3.2")
                elif "asyncpg" in str(e):
                    print(f"   Solution: pip install asyncpg==0.29.0")
        except Exception as e:
            services['event_similarity'] = False
            services['event_similarity_service'] = None
            if verbose:
                if "ufunc" in str(e) or "scipy" in str(e):
                    print(f"❌ Event Similarity service - scipy/numpy compatibility issue")
                    print(f"   Solution: pip install scipy==1.10.1 numpy==1.24.3")
                else:
                    print(f"❌ Event Similarity service error: {e}")
                    logger.exception("Detailed Event Similarity service error:")

        return services

    except Exception as e:
        if verbose:
            print(f"❌ Service initialization failed: {e}")
        logger.exception("Service initialization failed:")
        return {}


def validate_critical_services(services: Dict[str, Any], verbose: bool = True) -> bool:
    """
    Validate that critical services are ready

    Args:
        services: Dictionary of services
        verbose: Whether to print verbose output

    Returns:
        True if all critical services are ready, False otherwise
    """
    critical_services = ['places365', 'voyage', 'claude', 'database', 'event_similarity']
    missing_services = [s for s in critical_services if not services.get(s, False)]

    if missing_services:
        if verbose:
            print(f"\n⚠️ Warning: Some services are not available: {missing_services}")
            print("The test may fail or produce incomplete results.")

        if 'event_similarity' in missing_services:
            if verbose:
                print("❌ Event Similarity service is required. Cannot proceed.")
            return False

    if verbose:
        print(f"\n✅ All services initialized successfully")

    return True
