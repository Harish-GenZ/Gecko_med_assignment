import math
from collections.abc import Sequence


def normalize_vector(vector: Sequence[float]) -> list[float]:
    """
    Computes L2 (Euclidean) normalized vector for cosine similarity calculations.
    Returns unit vector where ||v||_2 = 1.0.
    """
    if not vector:
        raise ValueError("Cannot normalize an empty vector.")

    norm = math.sqrt(sum(float(x) ** 2 for x in vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-magnitude vector.")

    return [float(x) / norm for x in vector]


def cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """
    Computes cosine similarity between two numeric vectors: (u . v) / (||u|| * ||v||).
    Returns a float in [-1.0, 1.0].
    """
    if len(vec1) != len(vec2):
        raise ValueError(
            f"Vector dimension mismatch: len(vec1)={len(vec1)} vs len(vec2)={len(vec2)}"
        )

    dot_product = sum(float(a) * float(b) for a, b in zip(vec1, vec2, strict=False))
    norm_a = math.sqrt(sum(float(a) ** 2 for a in vec1))
    norm_b = math.sqrt(sum(float(b) ** 2 for b in vec2))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    sim = dot_product / (norm_a * norm_b)
    # Clamp to [-1.0, 1.0] to account for minor floating-point precision inaccuracies
    return max(-1.0, min(1.0, float(sim)))


def validate_name_text(text_val: str | None) -> str:
    """
    Validates outlet name input for embedding generation.
    Rejects empty, whitespace-only, or non-string inputs.
    """
    if text_val is None:
        raise ValueError("Outlet name cannot be None.")
    if not isinstance(text_val, str):
        raise ValueError(f"Outlet name must be a string, got {type(text_val).__name__}.")

    cleaned = text_val.strip()
    if not cleaned:
        raise ValueError("Outlet name cannot be empty or whitespace-only.")

    return cleaned
