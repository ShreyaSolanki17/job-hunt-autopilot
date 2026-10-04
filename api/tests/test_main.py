import numpy as np
from fastapi.testclient import TestClient

from app.main import app, get_encoder, get_resume

client = TestClient(app)


class SameVector:
    def encode(self, sentences: list[str]) -> np.ndarray:
        return np.ones((len(sentences), 3))


def setup_function():
    app.dependency_overrides[get_encoder] = lambda: SameVector()
    app.dependency_overrides[get_resume] = lambda: "Python and Docker"


def teardown_function():
    app.dependency_overrides.clear()


BODY = {"job": {"job_id": "1", "title": "Dev", "description": "x"}, "required_skills": ["Python", "Rust"]}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_score():
    r = client.post("/score", json=BODY).json()
    assert (r["semantic_score"], r["keyword_score"], r["score"]) == (100, 50, 80)
    assert r["matched_skills"] == ["Python"] and r["gaps"] == ["Rust"]


def test_secret(monkeypatch):
    monkeypatch.setenv("API_SHARED_SECRET", "s")
    assert client.post("/score", json=BODY).status_code == 401
    assert client.post("/score", json=BODY, headers={"X-API-Secret": "s"}).status_code == 200
