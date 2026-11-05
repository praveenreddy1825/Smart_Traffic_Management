"""
Unit tests for the FastAPI Presentation Dashboard Server & Telemetry API (Module 11).
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
    assert isinstance(data, dict)


def test_api_scenarios(client):
    """Verify that /api/scenarios returns a list of curated scenario clips."""
    resp = client.get("/api/scenarios")
    assert resp.status_code == 200
    scenarios = resp.json()
    assert isinstance(scenarios, list)
    assert len(scenarios) > 0
    assert "file_name" in scenarios[0]


def test_control_mode(client):
    """Verify mode switching API."""
    resp = client.post("/api/control/mode", json={"mode": "crowd"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "crowd"
    assert pipeline.active_mode == "crowd"

    # Reset back to auto
    resp_auto = client.post("/api/control/mode", json={"mode": "auto"})
    assert resp_auto.status_code == 200
    assert pipeline.active_mode == "auto"


def test_control_injection(client):
    """Verify emergency vehicle beacon injection control endpoints."""
    # Inject emergency
    resp_inject = client.post("/api/control/inject", json={"has_emergency": True, "vehicle_type": "Ambulance"})
    assert resp_inject.status_code == 200
    assert resp_inject.json()["injected"] is True

    assert pipeline.injected_emergency is True
    assert pipeline.injected_label == "Ambulance"

    # Clear injection
    resp_clear = client.post("/api/control/clear-inject")
    assert resp_clear.status_code == 200
    assert resp_clear.json()["injected"] is False
    assert pipeline.injected_emergency is False


