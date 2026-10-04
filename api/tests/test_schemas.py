import pytest
from pydantic import ValidationError

from app.schemas import MAX_DESCRIPTION_CHARS, AnalysisOut, JobIn

OUT = dict(
    job_id="a-1", score=87, semantic_score=82, keyword_score=94,
    matched_skills=["Python"], gaps=[], summary="ok",
    email_subject="s", email_draft="hi", skipped_email=False,
)


def test_job_in_defaults_and_empty_description():
    job = JobIn(job_id="a-1", title="AI Engineer")
    assert job.description == ""


def test_job_in_requires_id_and_title():
    with pytest.raises(ValidationError):
        JobIn(job_id="", title="x")
    with pytest.raises(ValidationError):
        JobIn(job_id="a")


def test_job_in_truncates_long_description():
    job = JobIn(job_id="a", title="t", description="x" * (MAX_DESCRIPTION_CHARS + 500))
    assert len(job.description) == MAX_DESCRIPTION_CHARS


def test_job_in_non_english():
    assert JobIn(job_id="a", title="エンジニア", description="機械学習 Ingénieur").title == "エンジニア"


def test_analysis_out_valid():
    assert AnalysisOut(**OUT).score == 87


@pytest.mark.parametrize("field", ["score", "semantic_score", "keyword_score"])
@pytest.mark.parametrize("bad", [-1, 101])
def test_analysis_out_score_bounds(field, bad):
    with pytest.raises(ValidationError):
        AnalysisOut(**{**OUT, field: bad})


def test_analysis_out_missing_field():
    data = {k: v for k, v in OUT.items() if k != "summary"}
    with pytest.raises(ValidationError):
        AnalysisOut(**data)


def test_skipped_email_requires_empty_draft():
    with pytest.raises(ValidationError):
        AnalysisOut(**{**OUT, "skipped_email": True})
    assert AnalysisOut(**{**OUT, "skipped_email": True, "email_draft": ""}).skipped_email
