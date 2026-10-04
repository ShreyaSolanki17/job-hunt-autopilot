from typing import Annotated, Self

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_DESCRIPTION_CHARS = 8000  # safe token budget for the LLM and embedder

Score = Annotated[int, Field(ge=0, le=100)]


class JobIn(BaseModel):
    job_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    company: str = ""
    location: str = ""
    description: str = ""
    url: str = ""
    source: str = ""

    @field_validator("description")
    @classmethod
    def truncate_description(cls, v: str) -> str:
        return v[:MAX_DESCRIPTION_CHARS]


class AnalysisOut(BaseModel):
    job_id: str
    score: Score
    semantic_score: Score
    keyword_score: Score
    matched_skills: list[str]
    gaps: list[str]
    summary: str
    email_subject: str
    email_draft: str
    skipped_email: bool

    @model_validator(mode="after")
    def skipped_means_no_draft(self) -> Self:
        if self.skipped_email and self.email_draft:
            raise ValueError("email_draft must be empty when skipped_email is true")
        return self


# Internal crew task outputs (used as output_pydantic).
class Requirements(BaseModel):
    required_skills: list[str]
    nice_to_haves: list[str] = []
    seniority: str = ""
    red_flags: list[str] = []


class FitExplanation(BaseModel):
    matched_skills: list[str]
    gaps: list[str]
    summary: str


class Email(BaseModel):
    subject: str
    body: str


class ScoreIn(BaseModel):
    job: JobIn
    required_skills: list[str] = []


class ScoreOut(BaseModel):
    job_id: str
    score: Score
    semantic_score: Score
    keyword_score: Score
    matched_skills: list[str]
    gaps: list[str]
