"""
Embedding services package for outlet name matching and photograph similarity.
"""

from app.services.embeddings.base import (
    cosine_similarity,
    normalize_vector,
    validate_name_text,
)
from app.services.embeddings.image_service import (
    ImageEmbeddingService,
    generate_image_embedding,
    get_image_embedding_service,
)
from app.services.embeddings.text_service import (
    TextEmbeddingService,
    generate_name_embedding,
    get_text_embedding_service,
)

__all__ = [
    "normalize_vector",
    "cosine_similarity",
    "validate_name_text",
    "TextEmbeddingService",
    "get_text_embedding_service",
    "generate_name_embedding",
    "ImageEmbeddingService",
    "get_image_embedding_service",
    "generate_image_embedding",
]
