import os
from collections.abc import Callable

from pydantic import BaseModel

from app import scoring
from app.schemas import AnalysisOut, Email, FitExplanation, JobIn, Requirements

EMAIL_MIN_SCORE = 50  # below this we skip drafting an email

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


def run_task(role: str, prompt: str, schema: type[BaseModel]) -> BaseModel:
    """Run one single-agent crew and return its structured output."""
    from crewai import Agent, Crew, Task

    llm = _llm()
    agent = Agent(role=role, goal=role, backstory=role, llm=llm, allow_delegation=False)
    task = Task(description=prompt, expected_output="Structured result", agent=agent, output_pydantic=schema)
    return Crew(agents=[agent], tasks=[task]).kickoff().pydantic


def analyze(job: JobIn, resume: str, encoder: scoring.Encoder, run: Runner = run_task) -> AnalysisOut:
    job_text = f"{job.title} at {job.company}\n{job.description}"

    reqs = run("Job requirements analyst", f"Extract requirements from this job:\n{job_text}", Requirements)
    matched, gaps = scoring.skill_matches(resume, reqs.required_skills)
    sem = scoring.semantic_score(resume, job_text, encoder)
    kw = scoring.keyword_score(resume, reqs.required_skills)
    score = scoring.final_score(sem, kw)

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
    )
