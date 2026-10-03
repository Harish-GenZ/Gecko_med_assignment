import base64
import logging
from typing import Any
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.retrieval import (
    CandidateRetrievalService,
    RetrievalResult,
    get_candidate_retrieval_service,
)

logger = logging.getLogger("outlet_verification.api.candidates")

router = APIRouter(prefix="/candidates", tags=["Candidate Retrieval"])


class CandidateSearchJsonRequest(BaseModel):
    """
    JSON request body for candidate retrieval testing.
    """
    name: str = Field(..., example="Apollo Pharmacy - Indiranagar")
    latitude: float = Field(..., ge=-90.0, le=90.0, example=12.9716)
    longitude: float = Field(..., ge=-180.0, le=180.0, example=77.5946)
    image_base64: str | None = Field(None, description="Base64 encoded photograph data")
    image_url: str | None = Field(None, description="Existing image URL or local path")
    name_top_k: int | None = Field(None, gt=0, example=5)
    image_top_k: int | None = Field(None, gt=0, example=5)
    geo_radius_meters: float | None = Field(None, gt=0, example=500.0)
    geo_top_k: int | None = Field(None, gt=0, example=10)


@router.post(
    "/search",
    response_model=RetrievalResult,
    summary="Search candidate outlets via multipart form",
    description="Surfaces candidate outlets via Name HNSW vector search, Image HNSW vector search, and Geographic radius pre-filtering.",
)
async def search_candidates_form(
    name: str = Form(..., description="Outlet name"),
    latitude: float = Form(..., description="Latitude coordinate (-90 to +90)"),
    longitude: float = Form(..., description="Longitude coordinate (-180 to +180)"),
    image: UploadFile = File(..., description="Outlet photograph file"),
    name_top_k: int | None = Form(None),
    image_top_k: int | None = Form(None),
    geo_radius_meters: float | None = Form(None),
    geo_top_k: int | None = Form(None),
    db: Session = Depends(get_db),
    retrieval_service: CandidateRetrievalService = Depends(get_candidate_retrieval_service),
) -> RetrievalResult:
    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image file is empty.",
            )

        return retrieval_service.retrieve_candidates(
            db=db,
            name=name,
            latitude=latitude,
            longitude=longitude,
            image=image_bytes,
            name_top_k=name_top_k,
            image_top_k=image_top_k,
            geo_radius_meters=geo_radius_meters,
            geo_top_k=geo_top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Error in candidate search endpoint: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Candidate retrieval failed: {exc}",
        ) from exc


@router.post(
    "/search-json",
    response_model=RetrievalResult,
    summary="Search candidate outlets via JSON payload",
    description="Development/testing endpoint accepting base64 image or image URL.",
)
def search_candidates_json(
    req: CandidateSearchJsonRequest,
    db: Session = Depends(get_db),
    retrieval_service: CandidateRetrievalService = Depends(get_candidate_retrieval_service),
) -> RetrievalResult:
    try:
        # Determine image input
        if req.image_base64:
            image_input: Any = base64.b64decode(req.image_base64)
        elif req.image_url:
            image_input = req.image_url
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'image_base64' or 'image_url' must be provided.",
            )

        return retrieval_service.retrieve_candidates(
            db=db,
            name=req.name,
            latitude=req.latitude,
            longitude=req.longitude,
            image=image_input,
            name_top_k=req.name_top_k,
            image_top_k=req.image_top_k,
            geo_radius_meters=req.geo_radius_meters,
            geo_top_k=req.geo_top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Error in candidate search-json endpoint: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Candidate retrieval failed: {exc}",
        ) from exc
