import itertools
import json
import logging
import re
import urllib.request
from collections.abc import Callable
from typing import Any

from app.schemas import JobIn

log = logging.getLogger(__name__)

# Public board names: boards-api.greenhouse.io/v1/boards/<token>, api.lever.co/v0/postings/<slug>
GREENHOUSE_BOARDS = ["reddit", "instacart", "samsara", "databricks", "anthropic"]
LEVER_COMPANIES = ["meesho", "outreach", "paytm", "zeta"]

TITLE = re.compile(r"\b(ai|ml|machine learning|data scientist|llm|nlp|deep learning|applied scientist|research engineer)\b", re.I)
NEAR_ME = re.compile(
    r"india|bangalore|bengaluru|hyderabad|mumbai|pune|delhi|gurgaon|gurugram|noida|chennai|ahmedabad"
    r"|worldwide|anywhere|apac|asia",
    re.I,
)
BARE_REMOTE = re.compile(r"\W*remote\W*", re.I)  # "Remote - US" is region-locked; only plain "Remote" passes
# Wrong level/function for a fresher AI/ML engineer search; also saves LLM quota.
EXCLUDE = re.compile(
    r"\b(manager|director|head|vp|sales|finance|gtm|pre-?sales|senior|sr|staff|principal|lead)\b", re.I
)
MAX_JOBS = 15  # caps LLM calls per run

GetJson = Callable[[str], Any]


def http_get(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "job-hunt-autopilot"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _wanted(title: str, location: str) -> bool:
    return bool(TITLE.search(title) and not EXCLUDE.search(title) and (NEAR_ME.search(location) or BARE_REMOTE.fullmatch(location)))


def fetch_remotive(get: GetJson) -> list[JobIn]:
    # Remotive terms: link back to the job url and credit Remotive; poll at most ~4x/day.
    data = get("https://remotive.com/api/remote-jobs?category=software-dev")
    return [
        JobIn(
            job_id=f"remotive-{j['id']}",
            title=j["title"],
            company=j["company_name"],
            location=j.get("candidate_required_location", ""),
            description=j.get("description", ""),
            url=j["url"],
            source="remotive",
        )
        for j in data["jobs"]
        if _wanted(j["title"], j.get("candidate_required_location", ""))
    ]


def fetch_greenhouse(get: GetJson, token: str) -> list[JobIn]:
    data = get(f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true")
    return [
        JobIn(
            job_id=f"greenhouse-{token}-{j['id']}",
            title=j["title"],
            company=j.get("company_name") or token,
            location=j["location"]["name"],
            description=j.get("content", ""),  # entity-escaped HTML; JobIn cleans it
            url=j["absolute_url"],
            source="greenhouse",
        )
        for j in data["jobs"]
        if _wanted(j["title"], j["location"]["name"])
    ]


def _lever_text(j: dict) -> str:
    # requirements usually sit in "lists", not in the intro
    lists = "\n".join(f"{x['text']}\n{x['content']}" for x in j.get("lists", []))
    return "\n".join([j.get("descriptionPlain", ""), lists, j.get("additionalPlain", "")])


def fetch_lever(get: GetJson, slug: str) -> list[JobIn]:
    jobs = []
    for j in get(f"https://api.lever.co/v0/postings/{slug}?mode=json"):
        loc = j.get("categories", {}).get("location") or ""
        if _wanted(j["text"], loc):
            jobs.append(
                JobIn(
                    job_id=f"lever-{slug}-{j['id']}",
                    title=j["text"],
                    company=slug.title(),
                    location=loc.strip(),
                    description=_lever_text(j),
                    url=j["hostedUrl"],
                    source="lever",
                )
            )
    return jobs


def fetch_all(get: GetJson = http_get) -> list[JobIn]:
    """All sources, filtered to AI/ML titles in India/remote. One failing source never blocks the rest."""
    calls = [lambda: fetch_remotive(get)]
    calls += [lambda t=t: fetch_greenhouse(get, t) for t in GREENHOUSE_BOARDS]
    calls += [lambda s=s: fetch_lever(get, s) for s in LEVER_COMPANIES]
    per_source: list[list[JobIn]] = []
    for call in calls:
        try:
            per_source.append(call())
        except Exception:
            log.warning("source failed", exc_info=True)
    # round-robin so the cap spreads across companies instead of keeping only the first one
    mixed = [j for group in itertools.zip_longest(*per_source) for j in group if j]
    return mixed[:MAX_JOBS]  # ponytail: no ranking or cross-run dedupe yet
