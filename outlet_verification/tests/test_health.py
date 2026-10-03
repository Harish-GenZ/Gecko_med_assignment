import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Outlet Verification System"
    assert data["status"] == "online"
    print("[PASS] Root endpoint test passed:", data)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "backend" in data
    assert data["backend"]["status"] == "running"
    assert "database" in data
    print("[PASS] Health endpoint test passed:", data)


if __name__ == "__main__":
    test_root()
    test_health()
    print("\nAll health tests passed successfully!")
