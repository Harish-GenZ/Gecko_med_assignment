import logging
import math
from typing import Any
from sqlalchemy.orm import Session

from app.config import get_settings
from app.services.embeddings import (
    generate_image_embedding,
    generate_name_embedding,
)
from app.services.retrieval.geo_retrieval import find_nearby_outlets
from app.services.retrieval.image_retrieval import search_by_image_embedding
from app.services.retrieval.models import CandidateEvidence, RetrievalResult
from app.services.retrieval.name_retrieval import search_by_name_embedding

logger = logging.getLogger("outlet_verification.retrieval.candidate_service")


def compute_haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Computes exact great-circle distance between two GPS coordinates in meters
    using the Haversine formula.
    """
    r_earth = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(r_earth * c, 2)


class CandidateRetrievalService:
    """
    Unified candidate retrieval engine.
    Orchestrates:
      1. Name semantic vector search (HNSW cosine)
      2. Image visual vector search (HNSW cosine)
      3. Geographic radius pre-filtering (earthdistance)
    Deduplicates and merges candidates into a unified candidate pool with multi-channel evidence.
    Explicitly does NOT assign duplicate labels, confidence scores, or classifications.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def retrieve_candidates(
        self,
        db: Session,
        name: str,
        latitude: float,
        longitude: float,
        image: Any,
        name_top_k: int | None = None,
        image_top_k: int | None = None,
        geo_radius_meters: float | None = None,
        geo_top_k: int | None = None,
    ) -> RetrievalResult:
        """
        Retrieves candidate outlets for a submission using multi-channel search.
        """
        # 1. Resolve search parameters (caller overrides or application settings)
        k_name = name_top_k if name_top_k is not None else self.settings.RETRIEVAL_NAME_TOP_K
        k_image = image_top_k if image_top_k is not None else self.settings.RETRIEVAL_IMAGE_TOP_K
        r_geo = (
            geo_radius_meters
            if geo_radius_meters is not None
            else self.settings.RETRIEVAL_GEO_RADIUS_METERS
        )
        k_geo = geo_top_k if geo_top_k is not None else self.settings.RETRIEVAL_GEO_TOP_K

        # 2. Generate query embeddings using existing Phase 3B services
        name_vector = generate_name_embedding(name) if name else None
        image_vector = generate_image_embedding(image) if image is not None else None

        # 3. Channel A: Name semantic vector search (if k_name > 0 and name_vector available)
        name_matches = (
            search_by_name_embedding(db, name_vector, top_k=k_name)
            if (k_name > 0 and name_vector is not None)
            else []
        )

        # 4. Channel B: Image visual vector search (if k_image > 0 and image_vector available)
        image_matches = (
            search_by_image_embedding(db, image_vector, top_k=k_image)
            if (k_image > 0 and image_vector is not None)
            else []
        )

        # 5. Channel C: Geographic radius search (if k_geo > 0 and r_geo > 0 and coordinates available)
        geo_matches = (
            find_nearby_outlets(
                db,
                latitude=latitude,
                longitude=longitude,
                radius_meters=r_geo,
                top_k=k_geo,
            )
            if (k_geo > 0 and r_geo > 0 and latitude is not None and longitude is not None)
            else []
        )

        # 6. Deduplicate & Merge Candidate Evidence by outlet_id
        candidates_map: dict[str, dict[str, Any]] = {}

        # Merge Name Matches
        for item in name_matches:
            oid = item["outlet_id"]
            if oid not in candidates_map:
                candidates_map[oid] = {
                    "outlet_id": oid,
                    "name": item["name"],
                    "latitude": item["latitude"],
                    "longitude": item["longitude"],
                    "image_url": item["image_url"],
                    "name_similarity": item["name_similarity"],
                    "image_similarity": None,
                    "distance_meters": None,
                    "matched_methods": ["name"],
                }
            else:
                candidates_map[oid]["name_similarity"] = item["name_similarity"]
                if "name" not in candidates_map[oid]["matched_methods"]:
                    candidates_map[oid]["matched_methods"].append("name")

        # Merge Image Matches
        for item in image_matches:
            oid = item["outlet_id"]
            if oid not in candidates_map:
                candidates_map[oid] = {
                    "outlet_id": oid,
                    "name": item["name"],
                    "latitude": item["latitude"],
                    "longitude": item["longitude"],
                    "image_url": item["image_url"],
                    "name_similarity": None,
                    "image_similarity": item["image_similarity"],
                    "distance_meters": None,
                    "matched_methods": ["image"],
                }
            else:
                candidates_map[oid]["image_similarity"] = item["image_similarity"]
                if "image" not in candidates_map[oid]["matched_methods"]:
                    candidates_map[oid]["matched_methods"].append("image")

        # Merge Geographic Matches
        for item in geo_matches:
            oid = item["outlet_id"]
            if oid not in candidates_map:
                candidates_map[oid] = {
                    "outlet_id": oid,
                    "name": item["name"],
                    "latitude": item["latitude"],
                    "longitude": item["longitude"],
                    "image_url": item["image_url"],
                    "name_similarity": None,
                    "image_similarity": None,
                    "distance_meters": item["distance_meters"],
                    "matched_methods": ["geo"],
                }
            else:
                candidates_map[oid]["distance_meters"] = item["distance_meters"]
                if "geo" not in candidates_map[oid]["matched_methods"]:
                    candidates_map[oid]["matched_methods"].append("geo")

        # For candidates retrieved via name/image that weren't in the geo_radius query,
        # compute their exact physical distance so the evidence profile is complete.
        for data in candidates_map.values():
            if (
                data["distance_meters"] is None
                and latitude is not None
                and longitude is not None
                and data["latitude"] is not None
                and data["longitude"] is not None
            ):
                data["distance_meters"] = compute_haversine_distance(
                    latitude, longitude, data["latitude"], data["longitude"]
                )

        # 7. Format into structured CandidateEvidence objects
        candidate_objects: list[CandidateEvidence] = [
            CandidateEvidence(**data) for data in candidates_map.values()
        ]

        # 8. Sort candidates deterministically:
        # Priority: number of matching channels (e.g. 3 channels > 2 channels > 1 channel),
        # then distance or similarity.
        candidate_objects.sort(
            key=lambda c: (
                len(c.matched_methods),
                (c.name_similarity or 0.0) + (c.image_similarity or 0.0),
                -(c.distance_meters if c.distance_meters is not None else 999999.0),
            ),
            reverse=True,
        )

        return RetrievalResult(
            total_candidates=len(candidate_objects),
            candidates=candidate_objects,
            search_parameters={
                "name_top_k": k_name,
                "image_top_k": k_image,
                "geo_radius_meters": r_geo,
                "geo_top_k": k_geo,
            },
        )


def get_candidate_retrieval_service() -> CandidateRetrievalService:
    """Factory getter for CandidateRetrievalService."""
    return CandidateRetrievalService()
