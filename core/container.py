"""
Service Container

Provides unified service management with singleton pattern and dependency injection.
"""

import logging
from typing import Dict, Any, Optional, Callable
from threading import Lock

logger = logging.getLogger(__name__)


class ServiceContainer:
    """
    Singleton service container for managing application services.

    Features:
    - Singleton pattern ensures one instance
    - Lazy loading of services
    - Service lifecycle management
    - Easy mocking for tests
    """

    _instance: Optional['ServiceContainer'] = None
    _lock: Lock = Lock()

    def __init__(self):
        if ServiceContainer._instance is not None:
            raise RuntimeError("ServiceContainer is a singleton. Use get_instance() instead.")

        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._initialized: Dict[str, bool] = {}
        logger.info("✅ ServiceContainer initialized")

    @classmethod
    def get_instance(cls) -> 'ServiceContainer':
        """Get or create singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = ServiceContainer()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset container (useful for testing)"""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.clear()
            cls._instance = None
        logger.info("🔄 ServiceContainer reset")

    def register(self, name: str, service: Any, factory: Optional[Callable] = None):
        """
        Register a service instance or factory

        Args:
            name: Service identifier
            service: Service instance (can be None if factory provided)
            factory: Optional factory function to create service lazily
        """
        if factory:
            self._factories[name] = factory
            self._initialized[name] = False
            logger.debug(f"📝 Registered factory for service: {name}")
        else:
            self._services[name] = service
            self._initialized[name] = True
            logger.debug(f"📝 Registered service: {name}")

    def register_factory(self, name: str, factory: Callable):
        """Register a factory function for lazy service creation"""
        self.register(name, None, factory)

    def get(self, name: str, default: Any = None) -> Any:
        """
        Get service by name (creates if factory exists)

        Args:
            name: Service identifier
            default: Default value if service not found

        Returns:
            Service instance or default
        """
        # Check if already instantiated
        if name in self._services and self._initialized.get(name, False):
            return self._services[name]

        # Try to create from factory
        if name in self._factories and not self._initialized.get(name, False):
            try:
                logger.debug(f"🏭 Creating service from factory: {name}")
                service = self._factories[name]()
                self._services[name] = service
                self._initialized[name] = True
                logger.info(f"✅ Service created: {name}")
                return service
            except Exception as e:
                logger.error(f"❌ Failed to create service '{name}': {e}")
                raise

        # Service not found
        if default is not None:
            return default

        raise KeyError(f"Service '{name}' not found in container")

    def has(self, name: str) -> bool:
        """Check if service is registered"""
        return name in self._services or name in self._factories

    def is_initialized(self, name: str) -> bool:
        """Check if service is initialized"""
        return self._initialized.get(name, False)

    def remove(self, name: str):
        """Remove service from container"""
        self._services.pop(name, None)
        self._factories.pop(name, None)
        self._initialized.pop(name, None)
        logger.debug(f"🗑️ Removed service: {name}")

    def clear(self):
        """Clear all services"""
        self._services.clear()
        self._factories.clear()
        self._initialized.clear()
        logger.info("🧹 ServiceContainer cleared")

    def list_services(self) -> Dict[str, bool]:
        """List all registered services and their initialization status"""
        all_services = set(self._services.keys()) | set(self._factories.keys())
        return {
            name: self._initialized.get(name, False)
            for name in all_services
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get container statistics"""
        return {
            'total_registered': len(self.list_services()),
            'initialized': sum(1 for v in self._initialized.values() if v),
            'pending': sum(1 for v in self._initialized.values() if not v),
            'services': self.list_services()
        }


# Convenience function
def get_container() -> ServiceContainer:
    """Get the service container instance"""
    return ServiceContainer.get_instance()


# Service name constants
class ServiceNames:
    """Constants for service names"""
    PLACES365 = "places365"
    VOYAGE_CLIENT = "voyage_client"
    CLAUDE = "claude"
    DATABASE = "database"
    EVENT_SIMILARITY = "event_similarity"
    ABOUT_EXTRACTION = "about_extraction"
    PROMPT_MANAGER = "prompt_manager"


__all__ = [
    'ServiceContainer',
    'get_container',
    'ServiceNames'
]
