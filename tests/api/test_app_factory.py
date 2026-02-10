"""Tests for the app factory pattern in api.main.

Demonstrates how create_app() enables creating test apps with subsets of routers
for better test isolation and faster test execution.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.main import create_app


def test_create_app_returns_fastapi_instance():
    """create_app() returns a properly configured FastAPI instance."""
    app = create_app()

    assert isinstance(app, FastAPI)
    assert app.title == "JARVIS API"
    assert app.version == "1.0.0"


def test_create_app_includes_all_routers():
    """create_app() includes all expected routers."""
    app = create_app()

    # Get all route paths
    route_paths = {route.path for route in app.routes}

    # Verify core endpoints are present
    assert any(path.startswith("/health") for path in route_paths)
    assert any(path.startswith("/conversations") for path in route_paths)
    assert any(path.startswith("/drafts") for path in route_paths)


def test_create_app_configures_middleware():
    """create_app() configures middleware correctly."""
    app = create_app()

    # Test CORS middleware is configured
    client = TestClient(app)
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})

    # CORS headers should be present
    assert "access-control-allow-origin" in response.headers


def test_create_app_configures_rate_limiting():
    """create_app() configures rate limiting."""
    app = create_app()

    # Limiter should be in app state
    assert hasattr(app.state, "limiter")
    assert app.state.limiter is not None


def test_create_app_multiple_instances_are_independent():
    """Multiple calls to create_app() return independent instances."""
    app1 = create_app()
    app2 = create_app()

    # Should be different instances
    assert app1 is not app2

    # But should have same configuration
    assert app1.title == app2.title
    assert app1.version == app2.version


def test_minimal_test_app_example():
    """Example: Creating a minimal test app with only health router for fast tests."""
    # This demonstrates how you could create a minimal app for specific tests
    # (though for this example we just verify the pattern works)

    from fastapi import FastAPI

    from api.routers.health import router as health_router

    # Create a minimal FastAPI app for testing
    minimal_app = FastAPI()
    minimal_app.include_router(health_router)

    # Verify it works
    client = TestClient(minimal_app)
    response = client.get("/health")
    assert response.status_code == 200

    # This minimal app loads much faster than the full app
    # and reduces test coupling for health-only tests
