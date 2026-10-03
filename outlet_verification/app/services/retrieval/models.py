from typing import Any
from pydantic import BaseModel, Field


class CandidateEvidence(BaseModel):
    """
    Evidence profile for an existing outlet identified during candidate retrieval.
    Captures multi-channel similarities and geographic proximity.
    Explicitly does NOT assign duplicate or confidence classifications.
    """
    outlet_id: str = Field(..., description="Unique UUID of the existing candidate outlet")
    name: str = Field(..., description="Name of the existing candidate outlet")
    latitude: float = Field(..., description="Latitude of the candidate outlet")
    longitude: float = Field(..., description="Longitude of the candidate outlet")
    image_url: str = Field(..., description="Photograph URL of the candidate outlet")
    
    # Evidence metrics (None if not retrieved or not applicable in a channel)
    name_similarity: float | None = Field(
        default=None, 
        description="Cosine similarity between names [0.0 to 1.0] from pgvector HNSW search"
    )
    image_similarity: float | None = Field(
        default=None, 
        description="Cosine similarity between photographs [0.0 to 1.0] from pgvector HNSW search"
    )
    distance_meters: float | None = Field(
        default=None, 
        description="Physical great-circle distance in meters"
    )
    
    matched_methods: list[str] = Field(
        default_factory=list, 
        description="List of retrieval channels that surfaced this candidate (e.g., ['name', 'geo'])"
    )


class RetrievalResult(BaseModel):
    """
    Aggregate response returned by candidate retrieval service.
    """
    total_candidates: int
    candidates: list[CandidateEvidence]
    search_parameters: dict[str, Any]
