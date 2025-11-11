"""
Unit Tests for ServiceContainer

Tests for service container and dependency injection.
"""

import pytest
from core.container import ServiceContainer, ServiceNames


class TestServiceContainer:
    """Test ServiceContainer class"""

    def setup_method(self):
        """Reset container before each test"""
        ServiceContainer.reset()

    def teardown_method(self):
        """Clean up after each test"""
        ServiceContainer.reset()

    def test_singleton_pattern(self):
        """Test that container follows singleton pattern"""
        container1 = ServiceContainer.get_instance()
        container2 = ServiceContainer.get_instance()
        assert container1 is container2

    def test_register_service(self):
        """Test registering a service"""
        container = ServiceContainer.get_instance()

        # Create a mock service
        class MockService:
            def __init__(self):
                self.value = "test"

        service = MockService()
        container.register("mock", service)

        # Retrieve and verify
        retrieved = container.get("mock")
        assert retrieved is service
        assert retrieved.value == "test"

    def test_register_factory(self):
        """Test registering a service factory"""
        container = ServiceContainer.get_instance()

        # Create a factory
        def create_service():
            class MockService:
                def __init__(self):
                    self.value = "factory_test"
            return MockService()

        container.register_factory("mock", create_service)

        # Service should not be initialized yet
        assert not container.is_initialized("mock")

        # Get service (should trigger factory)
        service = container.get("mock")
        assert service.value == "factory_test"

        # Now should be initialized
        assert container.is_initialized("mock")

    def test_get_nonexistent_service_raises_error(self):
        """Test getting nonexistent service raises KeyError"""
        container = ServiceContainer.get_instance()

        with pytest.raises(KeyError) as exc_info:
            container.get("nonexistent")

        assert "not found" in str(exc_info.value)

    def test_get_with_default(self):
        """Test getting service with default value"""
        container = ServiceContainer.get_instance()

        result = container.get("nonexistent", default="default_value")
        assert result == "default_value"

    def test_has_service(self):
        """Test checking if service exists"""
        container = ServiceContainer.get_instance()

        class MockService:
            pass

        container.register("mock", MockService())

        assert container.has("mock")
        assert not container.has("nonexistent")

    def test_remove_service(self):
        """Test removing a service"""
        container = ServiceContainer.get_instance()

        class MockService:
            pass

        container.register("mock", MockService())
        assert container.has("mock")

        container.remove("mock")
        assert not container.has("mock")

    def test_clear_container(self):
        """Test clearing all services"""
        container = ServiceContainer.get_instance()

        container.register("service1", "value1")
        container.register("service2", "value2")

        assert container.has("service1")
        assert container.has("service2")

        container.clear()

        assert not container.has("service1")
        assert not container.has("service2")

    def test_list_services(self):
        """Test listing all services"""
        container = ServiceContainer.get_instance()

        container.register("service1", "value1")
        container.register_factory("service2", lambda: "value2")

        services = container.list_services()

        assert "service1" in services
        assert "service2" in services
        assert services["service1"] is True  # Initialized
        assert services["service2"] is False  # Not initialized yet

    def test_get_statistics(self):
        """Test getting container statistics"""
        container = ServiceContainer.get_instance()

        container.register("service1", "value1")
        container.register_factory("service2", lambda: "value2")

        stats = container.get_statistics()

        assert stats['total_registered'] == 2
        assert stats['initialized'] == 1
        assert stats['pending'] == 1

    def test_factory_called_only_once(self):
        """Test that factory is called only once"""
        container = ServiceContainer.get_instance()

        call_count = {'count': 0}

        def create_service():
            call_count['count'] += 1
            return f"service_{call_count['count']}"

        container.register_factory("mock", create_service)

        # Get service multiple times
        service1 = container.get("mock")
        service2 = container.get("mock")
        service3 = container.get("mock")

        # Factory should be called only once
        assert call_count['count'] == 1

        # All returns should be the same instance
        assert service1 is service2
        assert service2 is service3


class TestServiceNames:
    """Test ServiceNames constants"""

    def test_service_names_defined(self):
        """Test that service names are defined"""
        assert hasattr(ServiceNames, 'PLACES365')
        assert hasattr(ServiceNames, 'VOYAGE_CLIENT')
        assert hasattr(ServiceNames, 'CLAUDE')
        assert hasattr(ServiceNames, 'DATABASE')
        assert hasattr(ServiceNames, 'EVENT_SIMILARITY')
