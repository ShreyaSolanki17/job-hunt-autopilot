import logging
import os
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel

from app import scoring
from app.schemas import AnalysisOut, Email, FitExplanation, JobIn, Requirements

log = logging.getLogger(__name__)
T = TypeVar("T")

EMAIL_MIN_SCORE = 50  # below this we skip drafting an email
MAX_YEARS_EXPERIENCE = 1  # jobs asking for more years than this are filtered out

# (role, instructions, schema) -> parsed schema instance
Runner = Callable[[str, str, type[BaseModel]], BaseModel]

DEFAULT_MODELS = {"groq": "groq/openai/gpt-oss-120b", "gemini": "gemini/gemini-2.0-flash"}


def _llm():
    from crewai import LLM

    provider = os.getenv("LLM_PROVIDER", "groq")
    return LLM(model=os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider])


def warm_up() -> None:
    """Load crewai + litellm (slow, lazily imported) once at startup.

    Doing it on the first request lets two concurrent requests race each other
    inside the import and fail with KeyError: 'litellm'.
    """
    _llm()


def _retry(fn: Callable[[], T], attempts: int = 3) -> T:
    """Groq sometimes answers with valid JSON as plain text and rejects it (tool_use_failed); a retry usually works."""
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            log.warning("LLM call failed (attempt %d/%d), retrying", i + 1, attempts, exc_info=True)


def run_task(role: str, prompt: str, schema: type[BaseModel]) -> BaseModel:
    """Run one single-agent crew and return its structured output."""
    from crewai import Agent, Crew, Task

    def once() -> BaseModel:
        agent = Agent(role=role, goal=role, backstory=role, llm=_llm(), allow_delegation=False)
        task = Task(description=prompt, expected_output="Structured result", agent=agent, output_pydantic=schema)
        return Crew(agents=[agent], tasks=[task]).kickoff().pydantic

    return _retry(once)


def analyze(job: JobIn, resume: str, encoder: scoring.Encoder, run: Runner = run_task) -> AnalysisOut:
    job_text = f"{job.title} at {job.company}\n{job.description}"

    reqs = run(
        "Job requirements analyst",
        "Extract requirements from this job. required_skills and nice_to_haves must be short skill, tool or "
        'technology names only (1-3 words each, e.g. "Python", "Docker", "RAG", "PyTorch"). Never write '
        "sentences, years of experience, degrees or certifications; leave those out of the skill lists. "
        "Separately, set min_years_experience to the minimum years of professional experience the job "
        'explicitly requires (take the lowest number of a range, e.g. "3-5 years" -> 3), or null if none is stated.\n'
        f"Job:\n{job_text}",
        Requirements,
    )
    matched, gaps = scoring.skill_matches(resume, reqs.required_skills)
    sem = scoring.semantic_score(resume, job_text, encoder)
    kw = scoring.keyword_score(resume, reqs.required_skills)
    score = scoring.final_score(sem, kw)

    years = reqs.min_years_experience
    if years is not None and years > MAX_YEARS_EXPERIENCE:
        # skip the fit and email LLM calls for jobs we would filter out anyway
        return AnalysisOut(
            job_id=job.job_id,
            score=score,
            semantic_score=sem,
            keyword_score=kw,
            matched_skills=matched,
            gaps=gaps,
            summary=f"Requires {years}+ years of experience.",
            email_subject="",
            email_draft="",
            skipped_email=True,
            min_years_experience=years,
            too_senior=True,
        )

    fit = run(
        "Career coach",
        f"Resume:\n{resume}\n\nJob:\n{job_text}\n\nMatched: {matched}\nGaps: {gaps}\n"
        "Explain the fit in 2-3 sentences, listing matched skills and gaps.",
        FitExplanation,
    )

    email = None
    if score >= EMAIL_MIN_SCORE:
        email = run(
            "Job application writer",
            f"Write a short, honest application email for this job.\nResume:\n{resume}\n\nJob:\n{job_text}\n"
            f"Fit: {fit.summary}",
            Email,
        )

    return AnalysisOut(
        job_id=job.job_id,
        score=score,
        semantic_score=sem,
        keyword_score=kw,
        matched_skills=matched,  # computed in Python, not trusted from the LLM
        gaps=gaps,
        summary=fit.summary,
        email_subject=email.subject if email else "",
        email_draft=email.body if email else "",
        skipped_email=email is None,
        min_years_experience=years,
    )
