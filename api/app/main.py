import os
from collections.abc import Callable
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException

from app import crew, scoring, sources
from app.schemas import AnalysisOut, JobIn, ScoreIn, ScoreOut

app = FastAPI(title="Job Hunt Autopilot")

REPO_ROOT = Path(__file__).resolve().parents[2]


def require_secret(x_api_secret: str = Header(default="")) -> None:
    secret = os.getenv("API_SHARED_SECRET", "")
    if secret and x_api_secret != secret:
        raise HTTPException(401, "bad or missing X-API-Secret")


def get_resume() -> str:
    path = REPO_ROOT / os.getenv("RESUME_PATH", "data/resume.md")
    if not path.exists():
        path = REPO_ROOT / "data" / "resume.sample.md"  # ponytail: sample fallback for fresh clones
    return path.read_text(encoding="utf-8")


def get_encoder() -> scoring.Encoder:
    return scoring.get_model()


def get_runner() -> crew.Runner:
    return crew.run_task


def get_fetcher() -> Callable[[], list[JobIn]]:
    return sources.fetch_all


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score", response_model=ScoreOut, dependencies=[Depends(require_secret)])
def score(
    req: ScoreIn,
    resume: str = Depends(get_resume),
    encoder: scoring.Encoder = Depends(get_encoder),
) -> ScoreOut:
    job_text = f"{req.job.title}\n{req.job.description}"
    sem = scoring.semantic_score(resume, job_text, encoder)
    kw = scoring.keyword_score(resume, req.required_skills)
    matched, gaps = scoring.skill_matches(resume, req.required_skills)
    return ScoreOut(
        job_id=req.job.job_id,
        score=scoring.final_score(sem, kw),
        semantic_score=sem,
        keyword_score=kw,
        matched_skills=matched,
        gaps=gaps,
    )


@app.post("/analyze", response_model=AnalysisOut, dependencies=[Depends(require_secret)])
def analyze(
    job: JobIn,
    resume: str = Depends(get_resume),
    encoder: scoring.Encoder = Depends(get_encoder),
    run: crew.Runner = Depends(get_runner),
) -> AnalysisOut:
    return crew.analyze(job, resume, encoder, run)


@app.get("/jobs", response_model=list[JobIn], dependencies=[Depends(require_secret)])
def jobs(fetch: Callable[[], list[JobIn]] = Depends(get_fetcher)) -> list[JobIn]:
    return fetch()
