import re
from functools import lru_cache
from typing import Protocol

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

# Each group is a set of interchangeable spellings.
ALIAS_GROUPS: list[set[str]] = [
    {"js", "javascript"},
    {"ts", "typescript"},
    {"ml", "machine learning"},
    {"dl", "deep learning"},
    {"nlp", "natural language processing"},
    {"llm", "llms", "large language models", "large language model"},
    {"k8s", "kubernetes"},
    {"postgres", "postgresql"},
    {"sklearn", "scikit-learn", "scikit learn"},
    {"torch", "pytorch"},
    {"huggingface", "hugging face"},
    {"genai", "gen ai", "generative ai"},
    {"gcp", "google cloud", "google cloud platform"},
    {"aws", "amazon web services"},
    {"rag", "retrieval augmented generation", "retrieval-augmented generation"},
    {"cv", "computer vision"},
]
_ALIASES = {term: group for group in ALIAS_GROUPS for term in group}


class Encoder(Protocol):
    def encode(self, sentences: list[str]) -> np.ndarray: ...


@lru_cache(maxsize=1)
def get_model() -> Encoder:
    """Load the embedding model once per process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def semantic_score(resume: str, job_text: str, model: Encoder | None = None) -> int:
    """Cosine similarity of embeddings, scaled to 0-100."""
    if not resume.strip() or not job_text.strip():
        return 0
    a, b = (model or get_model()).encode([resume, job_text])
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0
    return round(max(0.0, min(1.0, float(np.dot(a, b)) / denom)) * 100)


def _variants(skill: str) -> set[str]:
    s = skill.strip().lower()
    return _ALIASES.get(s, {s})


def _in_text(term: str, text: str) -> bool:
    # custom boundaries so "c++" / "c#" / "node.js" match and "java" != "javascript"
    return re.search(rf"(?<![\w+#]){re.escape(term)}(?![\w+#])", text) is not None


def skill_matches(resume: str, required_skills: list[str]) -> tuple[list[str], list[str]]:
    """Split required skills into (matched, missing), preserving input order."""
    text = resume.lower()
    skills = [s for s in required_skills if s.strip()]
    matched = [s for s in skills if any(_in_text(v, text) for v in _variants(s))]
    return matched, [s for s in skills if s not in matched]


def keyword_score(resume: str, required_skills: list[str]) -> int:
    """Percent of required skills found in the resume (case-insensitive, aliased)."""
    skills = [s for s in required_skills if s.strip()]
    if not skills:
        return 0
    matched, _ = skill_matches(resume, skills)
    return round(100 * len(matched) / len(skills))


def final_score(semantic: int, keyword: int) -> int:
    return round(0.6 * semantic + 0.4 * keyword)
