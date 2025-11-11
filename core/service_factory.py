"""
Service Factory

Consolidates service initialization logic into a single reusable module.
Used by both main.py and tests.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from config import config
from core.container import ServiceContainer, ServiceNames
from core.exceptions import ServiceUnavailableError, ConfigurationError

logger = logging.getLogger(__name__)


class ServiceFactory:
    """Factory for creating and initializing all application services"""

    @staticmethod
    async def initialize_all(
        container: Optional[ServiceContainer] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Initialize all services and register them in the container

        Args:
            container: ServiceContainer instance (creates new if None)
            verbose: Print detailed initialization info

        Returns:
            Dictionary with service status
        """
        if container is None:
            container = ServiceContainer.get_instance()

        results = {}

        if verbose:
            print("🔧 Initializing services...")

        # Initialize in dependency order
        try:
            # 1. Places365 Model
            results['places365'] = await ServiceFactory._init_places365(container, verbose)

            # 2. Voyage Client
            results['voyage'] = await ServiceFactory._init_voyage(container, verbose)

            # 3. Claude Service
            results['claude'] = await ServiceFactory._init_claude(container, verbose)

            # 4. Database Service
            results['database'] = await ServiceFactory._init_database(container, verbose)

            # 5. Prompt Manager
            results['prompt_manager'] = await ServiceFactory._init_prompt_manager(container, verbose)

            # 6. Event Similarity Service (depends on above)
            results['event_similarity'] = await ServiceFactory._init_event_similarity(container, verbose)

            # 7. About Extraction Service
            results['about_extraction'] = await ServiceFactory._init_about_extraction(container, verbose)

            if verbose:
                print("\n✅ All services initialized successfully")

        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
            if verbose:
                print(f"\n❌ Service initialization failed: {e}")
            raise

        return results

    @staticmethod
    async def _init_places365(container: ServiceContainer, verbose: bool) -> bool:
        """Initialize Places365 model"""
        if verbose:
            print("📥 Loading Places365 model...")

        try:
            from models.places365 import get_places365_model

            model = get_places365_model()
            container.register(ServiceNames.PLACES365, model)

            if verbose:
                status = "✅" if model is not None else "❌"
                print(f"{status} Places365 model: {'loaded' if model else 'failed'}")

            return model is not None

        except Exception as e:
            logger.error(f"Failed to load Places365 model: {e}")
            if verbose:
                print(f"❌ Places365 model failed: {e}")
            raise ServiceUnavailableError("Places365", str(e))

    @staticmethod
    async def _init_voyage(container: ServiceContainer, verbose: bool) -> bool:
        """Initialize Voyage client"""
        if verbose:
            print("🚀 Initializing Voyage client...")

        try:
            from utils.text_processor import get_voyage_client

            voyage_client = get_voyage_client()
            container.register(ServiceNames.VOYAGE_CLIENT, voyage_client)

            if verbose:
                status = "✅" if voyage_client else "❌"
                print(f"{status} Voyage client: {'initialized' if voyage_client else 'failed'}")

            return voyage_client is not None

        except Exception as e:
            logger.error(f"Failed to initialize Voyage client: {e}")
            if verbose:
                print(f"❌ Voyage client failed: {e}")
            raise ServiceUnavailableError("Voyage", str(e))

    @staticmethod
    async def _init_claude(container: ServiceContainer, verbose: bool) -> bool:
        """Initialize Claude service"""
        if verbose:
            print("🤖 Testing Claude service...")

        try:
            from services.claude_service import ClaudeService

            claude_service = ClaudeService()
            claude_status = claude_service.get_usage_stats()
            container.register(ServiceNames.CLAUDE, claude_service)

            if verbose:
                model = claude_status.get('model', 'unknown')
                print(f"✅ Claude service: ready ({model})")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize Claude service: {e}")
            if verbose:
                print(f"⚠️ Claude service: {e}")
                print("   Make sure CLAUDE_API_KEY is set in .env file")

            # Don't raise, just return False
            container.register(ServiceNames.CLAUDE, None)
            return False

    @staticmethod
    async def _init_database(container: ServiceContainer, verbose: bool) -> bool:
        """Initialize Database service"""
        if verbose:
            print("💾 Initializing Database service...")

        try:
            from services.database_service import DatabaseService

            db_service = DatabaseService()
            await db_service.initialize()

            # Test connection
            db_health = await db_service.health_check()
            is_healthy = db_health.get("status") == "healthy"

            container.register(ServiceNames.DATABASE, db_service)

            if verbose:
                status = "✅" if is_healthy else "❌"
                print(f"{status} Database service: {'connected' if is_healthy else 'failed'}")

            return is_healthy

        except Exception as e:
            logger.error(f"Failed to initialize Database service: {e}")
            if verbose:
                print(f"⚠️ Database service: {e}")

            container.register(ServiceNames.DATABASE, None)
            return False

    @staticmethod
    async def _init_prompt_manager(container: ServiceContainer, verbose: bool) -> bool:
        """Initialize Prompt Manager"""
        if verbose:
            print("📋 Initializing Prompt Manager...")

        try:
            from utils.prompt_manager import PromptManager

            # Check config files exist
            config_path = Path("config")
            required_configs = [
                "prompts.yaml",
                "output_formats.yaml",
                "similarity_prompts.yaml",
                "similarity_output_formats.yaml"
            ]

            missing_configs = [
                cfg for cfg in required_configs
                if not (config_path / cfg).exists()
            ]

            if missing_configs and verbose:
                print(f"⚠️ Missing config files: {missing_configs}")

            prompt_manager = PromptManager()
            container.register(ServiceNames.PROMPT_MANAGER, prompt_manager)

            if verbose:
                print("✅ Prompt Manager: initialized")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize Prompt Manager: {e}")
            if verbose:
                print(f"❌ Prompt Manager failed: {e}")
            raise ConfigurationError("Prompt Manager initialization failed", details={'error': str(e)})

    @staticmethod
    async def _init_event_similarity(container: ServiceContainer, verbose: bool) -> bool:
        """Initialize Event Similarity service"""
        if verbose:
            print("🔍 Initializing Event Similarity service...")

        try:
            from services.event_similarity_service import EventSimilarityService

            # Get dependencies
            claude_service = container.get(ServiceNames.CLAUDE)
            voyage_client = container.get(ServiceNames.VOYAGE_CLIENT)
            prompt_manager = container.get(ServiceNames.PROMPT_MANAGER)
            database_service = container.get(ServiceNames.DATABASE)

            # Check dependencies
            if not claude_service:
                raise ServiceUnavailableError("Claude service not available")
            if not voyage_client:
                raise ServiceUnavailableError("Voyage client not available")
            if not database_service:
                raise ServiceUnavailableError("Database service not available")

            # Create service
            event_similarity_service = EventSimilarityService(
                claude_service=claude_service,
                voyage_client=voyage_client,
                prompt_manager=prompt_manager,
                database_service=database_service
            )

            # Test service
            service_status = await event_similarity_service.get_service_status()
            is_operational = service_status.get("status") == "operational"

            container.register(ServiceNames.EVENT_SIMILARITY, event_similarity_service)

            if verbose:
                status = "✅" if is_operational else "❌"
                print(f"{status} Event Similarity service: {'ready' if is_operational else 'failed'}")

                if is_operational and verbose:
                    db_status = service_status.get('database_health', {})
                    faiss_stats = service_status.get('faiss_stats', {})
                    faiss_count = len([k for k, v in faiss_stats.items()
                                      if isinstance(v, dict) and v.get('total_vectors', 0) > 0])
                    print(f"   Database: {db_status.get('status')}")
                    print(f"   FAISS indexes: {faiss_count}")

            return is_operational

        except Exception as e:
            logger.error(f"Failed to initialize Event Similarity service: {e}")
            if verbose:
                print(f"❌ Event Similarity service: {e}")

            container.register(ServiceNames.EVENT_SIMILARITY, None)
            return False

    @staticmethod
    async def _init_about_extraction(container: ServiceContainer, verbose: bool) -> bool:
        """Initialize About Extraction service"""
        if verbose:
            print("📝 Initializing About Extraction service...")

        try:
            from services.about_extraction_service import AboutExtractionService

            about_service = AboutExtractionService()
            container.register(ServiceNames.ABOUT_EXTRACTION, about_service)

            if verbose:
                print("✅ About Extraction service: initialized")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize About Extraction service: {e}")
            if verbose:
                print(f"❌ About Extraction service: {e}")

            container.register(ServiceNames.ABOUT_EXTRACTION, None)
            return False

    @staticmethod
    def validate_critical_services(
        container: ServiceContainer,
        required_services: Optional[list] = None
    ) -> tuple[bool, list]:
        """
        Validate that critical services are initialized

        Args:
            container: ServiceContainer instance
            required_services: List of required service names (uses default if None)

        Returns:
            Tuple of (all_ready: bool, missing: list)
        """
        if required_services is None:
            required_services = [
                ServiceNames.PLACES365,
                ServiceNames.VOYAGE_CLIENT,
                ServiceNames.CLAUDE,
                ServiceNames.DATABASE,
                ServiceNames.EVENT_SIMILARITY
            ]

        missing = []
        for service_name in required_services:
            if not container.has(service_name):
                missing.append(service_name)
            elif not container.is_initialized(service_name):
                missing.append(service_name)
            elif container.get(service_name) is None:
                missing.append(service_name)

        return len(missing) == 0, missing


__all__ = ['ServiceFactory']
