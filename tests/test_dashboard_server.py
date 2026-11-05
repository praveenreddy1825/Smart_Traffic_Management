"""
Unit tests for the FastAPI Dashboard Server & Telemetry API (Module 11).
"""

import pytest
from starlette.testclient import TestClient

from app.dashboard.server import app, pipeline


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_index_page(client):
    """Verify that root URL returns the modern HTML dashboard."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "SmartCity Sentinel" in resp.text
    assert "North Approach" in resp.text


def test_api_health(client):
    """Verify that /api/health returns healthy status."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "device" in data


def test_api_status(client):
    """Verify that /api/status returns valid telemetry structure."""
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    # Check for core keys if pipeline has stepped or is initialized
    assert isinstance(data, dict)


def test_control_injection(client):
    """Verify emergency vehicle injection control endpoints."""
    # Inject emergency on North approach
    resp_inject = client.post("/api/control/inject", json={"direction": "north", "vehicle_type": "Ambulance"})
    assert resp_inject.status_code == 200
    assert resp_inject.json()["injected"] == "north"

    assert pipeline.injected_direction == "north"
    assert pipeline.injected_label == "Ambulance"

    # Clear injection
    resp_clear = client.post("/api/control/clear-inject")
    assert resp_clear.status_code == 200
    assert resp_clear.json()["injected"] is None
    assert pipeline.injected_direction is None


def test_invalid_direction_feed(client):
    """Verify that requesting an invalid feed direction returns 404."""
    resp = client.get("/video_feed/invalid_direction")
    assert resp.status_code == 404

