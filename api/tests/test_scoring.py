import numpy as np

from app.scoring import final_score, keyword_score, semantic_score, skill_matches

RESUME = "Python, FastAPI, PyTorch, RAG, JavaScript, C++, Docker, Machine Learning"


class FakeEncoder:
    """Maps texts to preset vectors so no model is downloaded."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def encode(self, sentences: list[str]) -> np.ndarray:
        return np.array([self.vectors[s] for s in sentences], dtype=float)


def test_keyword_full_and_partial():
    assert keyword_score(RESUME, ["Python", "docker"]) == 100
    assert keyword_score(RESUME, ["Python", "Kubernetes"]) == 50


def test_keyword_aliases():
    assert keyword_score(RESUME, ["js", "ml", "torch"]) == 100


def test_keyword_word_boundaries():
    assert keyword_score("I know JavaScript", ["Java"]) == 0
    assert keyword_score(RESUME, ["C++"]) == 100
    assert keyword_score("I know C", ["C++"]) == 0


def test_keyword_empty_inputs():
    assert keyword_score(RESUME, []) == 0
    assert keyword_score(RESUME, ["", "  "]) == 0
    assert keyword_score("", ["Python"]) == 0


def test_skill_matches_split():
    assert skill_matches(RESUME, ["python", "Kubernetes", "js"]) == (["python", "js"], ["Kubernetes"])


def test_semantic_identical_orthogonal_and_opposite():
    enc = FakeEncoder({"r": [1, 0], "same": [2, 0], "orth": [0, 1], "opp": [-1, 0]})
    assert semantic_score("r", "same", enc) == 100
    assert semantic_score("r", "orth", enc) == 0
    assert semantic_score("r", "opp", enc) == 0  # clipped, never negative


def test_semantic_empty_text_skips_model():
    assert semantic_score("", "job", model=None) == 0
    assert semantic_score("resume", "   ", model=None) == 0


def test_semantic_zero_vector():
    enc = FakeEncoder({"r": [1, 0], "z": [0, 0]})
    assert semantic_score("r", "z", enc) == 0


def test_final_score_weights():
    assert final_score(82, 94) == 87
    assert final_score(0, 0) == 0
    assert final_score(100, 100) == 100


def test_keyword_new_aliases():
    resume = "Hugging Face Transformers, Generative AI, Google Cloud, Retrieval-Augmented Generation, Computer Vision"
    assert keyword_score(resume, ["HuggingFace", "GenAI", "GCP", "RAG", "CV"]) == 100
    assert keyword_score("I know cvs and awsome tools", ["CV", "AWS"]) == 0  # no false hits inside other words
