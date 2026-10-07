from fastapi.testclient import TestClient

from app import sources
from app.main import app, get_fetcher

REMOTIVE = {"jobs": [
    {"id": 1, "title": "AI Engineer", "company_name": "A", "candidate_required_location": "Worldwide",
     "description": "<p>Python &amp; RAG</p>", "url": "https://remotive.com/x"},
    {"id": 2, "title": "AI Engineer", "company_name": "B", "candidate_required_location": "USA Only",
     "description": "x", "url": "u"},
    {"id": 4, "title": "Senior AI Engineer", "company_name": "D", "candidate_required_location": "Worldwide",
     "description": "x", "url": "u"},
    {"id": 5, "title": "AI Engineering Manager", "company_name": "E", "candidate_required_location": "Worldwide",
     "description": "x", "url": "u"},
    {"id": 3, "title": "Copywriter", "company_name": "C", "candidate_required_location": "Worldwide",
     "description": "x", "url": "u"},
]}
GREENHOUSE = {"jobs": [
    {"id": 7, "title": "ML Engineer", "company_name": "Acme", "location": {"name": "Bengaluru, India"},
     "content": "&lt;p&gt;Build&nbsp;models&lt;/p&gt;", "absolute_url": "https://gh/7"},
    {"id": 8, "title": "ML Engineer", "location": {"name": "New York"}, "content": "", "absolute_url": "u"},
    {"id": 9, "title": "ML Engineer", "location": {"name": "Remote - United States"}, "content": "", "absolute_url": "u"},
    {"id": 11, "title": "ML Engineer", "location": {"name": "Remote - Dallas"}, "content": "", "absolute_url": "u"},
    {"id": 12, "title": "ML Engineer", "location": {"name": "Remote"}, "content": "", "absolute_url": "u"},
    {"id": 10, "title": "ML Engineer", "location": {"name": "Remote - Europe, APAC"}, "content": "", "absolute_url": "u"},
]}
LEVER = [
    {"id": "abc", "text": "AI Researcher", "categories": {"location": "Mumbai"}, "descriptionPlain": "Intro",
     "lists": [{"text": "Requirements", "content": "<li>PyTorch</li>"}], "additionalPlain": "Perks",
     "hostedUrl": "https://lever/abc"},
    {"id": "def", "text": "Sales Lead", "categories": {"location": "Mumbai"}, "hostedUrl": "u"},
]


def fake_get(url):
    if "remotive" in url:
        return REMOTIVE
    if "greenhouse" in url:
        if "/reddit/" in url:
            raise OSError("board down")  # one bad source must not break the rest
        return GREENHOUSE
    return LEVER


def test_fetchers_filter_and_map(monkeypatch):
    assert [j.job_id for j in sources.fetch_remotive(fake_get)] == ["remotive-1"]
    assert sources.fetch_remotive(fake_get)[0].description == "Python & RAG"
    gh = sources.fetch_greenhouse(fake_get, "acme")
    assert [(j.job_id, j.description) for j in gh] == [("greenhouse-acme-7", "Build models"), ("greenhouse-acme-12", ""), ("greenhouse-acme-10", "")]
    lv = sources.fetch_lever(fake_get, "co")[0]
    assert lv.job_id == "lever-co-abc" and "PyTorch" in lv.description and "Perks" in lv.description


def test_level_and_function_blocklist():
    for title in ["Senior ML Engineer", "Staff Software Engineer - Machine Learning", "Sr. AI Engineer",
                  "Director of Applied Science", "Pre Sales - AI Solution Architect", "Finance Data and AI Lead",
                  "Principal Data Scientist", "AI Engineering Manager"]:
        assert not sources._wanted(title, "Bengaluru, India"), title
    for title in ["AI Engineer", "Machine Learning Engineer", "Data Scientist III", "Associate ML Engineer"]:
        assert sources._wanted(title, "Bengaluru, India"), title


def test_fetch_all_survives_failure_and_interleaves(monkeypatch):
    monkeypatch.setattr(sources, "GREENHOUSE_BOARDS", ["reddit", "acme"])
    monkeypatch.setattr(sources, "LEVER_COMPANIES", ["co"])
    ids = [j.job_id for j in sources.fetch_all(fake_get)]
    assert ids == ["remotive-1", "greenhouse-acme-7", "lever-co-abc", "greenhouse-acme-12", "greenhouse-acme-10"]  # interleaved
    assert not hasattr(sources, "MAX_JOBS")  # the per-run cap lives in n8n, after the seen filter


def test_jobs_endpoint():
    app.dependency_overrides[get_fetcher] = lambda: lambda: sources.fetch_remotive(fake_get)
    try:
        r = TestClient(app).get("/jobs")
        assert r.status_code == 200 and r.json()[0]["job_id"] == "remotive-1"
    finally:
        app.dependency_overrides.clear()


def test_data_roles_are_allowed():
    assert sources._wanted("Data Analyst", "Bengaluru, India")
    assert sources._wanted("Data Engineer", "Remote")
    assert not sources._wanted("Data Analyst", "Remote - US")
