"""
Candidate retrieval package for multi-channel outlet search.
"""

from app.services.retrieval.candidate_service import (
    CandidateRetrievalService,
    compute_haversine_distance,
    get_candidate_retrieval_service,
)
from app.services.retrieval.geo_retrieval import find_nearby_outlets
from app.services.retrieval.image_retrieval import search_by_image_embedding
from app.services.retrieval.models import CandidateEvidence, RetrievalResult
from app.services.retrieval.name_retrieval import search_by_name_embedding

__all__ = [
    "CandidateEvidence",
    "RetrievalResult",
    "find_nearby_outlets",
    "search_by_name_embedding",
    "search_by_image_embedding",
    "compute_haversine_distance",
    "CandidateRetrievalService",
    "get_candidate_retrieval_service",
]
