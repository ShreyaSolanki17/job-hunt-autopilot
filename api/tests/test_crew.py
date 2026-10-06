import numpy as np
from fastapi.testclient import TestClient

from app.crew import analyze
from app.main import app, get_encoder, get_resume, get_runner
from app.schemas import Email, FitExplanation, JobIn, Requirements

JOB = JobIn(job_id="1", title="Dev", company="Acme", description="x")
RESUME = "Python and Docker"


class SameVector:
    def encode(self, sentences: list[str]) -> np.ndarray:
        return np.ones((len(sentences), 3))


def fake_runner(skills: list[str]):
    out = {
        Requirements: Requirements(required_skills=skills),
        FitExplanation: FitExplanation(matched_skills=["llm says"], gaps=[], summary="Good fit"),
        Email: Email(subject="Hi", body="Hello"),
    }
    return lambda role, prompt, schema: out[schema]


def test_high_score_drafts_email():
    r = analyze(JOB, RESUME, SameVector(), fake_runner(["Python", "Docker"]))
    assert r.score == 100 and not r.skipped_email and r.email_draft == "Hello"
    assert r.matched_skills == ["Python", "Docker"]  # from Python, not the LLM


def test_low_score_skips_email():
    r = analyze(JOB, RESUME, type("Z", (), {"encode": lambda s, x: np.array([[1, 0], [0, 1]])})(), fake_runner(["Rust"]))
    assert r.score == 0 and r.skipped_email and r.email_draft == "" and r.email_subject == ""


def test_analyze_endpoint():
    app.dependency_overrides[get_encoder] = lambda: SameVector()
    app.dependency_overrides[get_resume] = lambda: RESUME
    app.dependency_overrides[get_runner] = lambda: fake_runner(["Python"])
    try:
        r = TestClient(app).post("/analyze", json=JOB.model_dump())
        assert r.status_code == 200 and r.json()["score"] == 100
    finally:
        app.dependency_overrides.clear()


def test_retry_recovers_then_gives_up():
    import pytest

    from app.crew import _retry

    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("tool_use_failed")
        return "ok"

    assert _retry(flaky) == "ok" and len(calls) == 3
    with pytest.raises(ValueError):
        _retry(lambda: (_ for _ in ()).throw(ValueError("always")), attempts=2)
