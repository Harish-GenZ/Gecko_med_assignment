import logging
from typing import Any
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger("outlet_verification.retrieval.name")

EXPECTED_NAME_DIMENSION = 384


def search_by_name_embedding(
    db: Session,
    embedding: list[float],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Retrieves top-K candidate outlets based on semantic name similarity
    using pgvector HNSW cosine distance index.
    Cosine similarity = 1.0 - (name_embedding <=> embedding).
    Excludes outlets with NULL name_embedding.
    Does NOT make duplicate/genuine decisions.
    """
    if not embedding:
        raise ValueError("Name embedding vector cannot be empty.")
    if len(embedding) != EXPECTED_NAME_DIMENSION:
        raise ValueError(
            f"Invalid name embedding dimension {len(embedding)}. Expected {EXPECTED_NAME_DIMENSION}."
        )
    if top_k <= 0:
        raise ValueError(f"top_k must be greater than zero, got {top_k}.")

    vec_str = f"[{','.join(str(float(x)) for x in embedding)}]"

    query = text("""
        SELECT 
            id,
            name,
            latitude,
            longitude,
            image_url,
            1.0 - (name_embedding <=> CAST(:query_vec AS vector)) AS similarity
        FROM outlets
        WHERE name_embedding IS NOT NULL
        ORDER BY name_embedding <=> CAST(:query_vec AS vector) ASC
        LIMIT :top_k;
    """)

    rows = db.execute(
        query,
        {
            "query_vec": vec_str,
            "top_k": int(top_k),
        },
    ).fetchall()

    candidates = []
    for row in rows:
        candidates.append({
            "outlet_id": str(row[0]),
            "name": str(row[1]),
            "latitude": float(row[2]),
            "longitude": float(row[3]),
            "image_url": str(row[4]),
            "name_similarity": round(float(row[5]), 4),
        })

    logger.debug("Name vector retrieval returned %d candidates.", len(candidates))
    return candidates
