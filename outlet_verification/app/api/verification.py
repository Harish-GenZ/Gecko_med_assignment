import base64
import logging
from typing import Any
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.storage_service import get_storage_service
from app.services.verification import (
    VerificationResponse,
    VerificationService,
    get_verification_service,
)

logger = logging.getLogger("outlet_verification.api.verification")

router = APIRouter(tags=["Verification Engine"])


class RegisterOutletResponse(BaseModel):
    """Response returned upon successful manual or direct outlet onboarding."""
    id: str = Field(..., description="UUID of registered outlet in PostgreSQL")
    name: str = Field(..., description="Outlet name")
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    image_url: str = Field(..., description="Storage URL of the uploaded image")
    message: str = Field("Outlet successfully registered in PostgreSQL and Railway Object Storage.")


class VerificationJsonRequest(BaseModel):
    """
    JSON request body for verification testing with base64 encoded photo or image URL.
    """
    name: str = Field(..., description="Name of the submitted outlet", example="Apollo Pharmacy - Indiranagar")
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude coordinate", example=12.9716)
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude coordinate", example=77.5946)
    image_base64: str | None = Field(None, description="Base64 encoded photograph data")
    image_url: str | None = Field(None, description="Existing photograph URL or local test image path")
    auto_register_if_genuine: bool = Field(False, description="Automatically register genuine outlets")
    name_top_k: int | None = Field(None, gt=0)
    image_top_k: int | None = Field(None, gt=0)
    geo_radius_meters: float | None = Field(None, gt=0)
    geo_top_k: int | None = Field(None, gt=0)


@router.post(
    "/verify",
    response_model=VerificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify submitted outlet via multipart form",
    description=(
        "Production verification endpoint. Computes embeddings, searches existing outlet database "
        "via multi-channel vector & spatial search, performs multimodal evidence fusion, and produces "
        "a deterministic DUPLICATE, GENUINE, or NEEDS_REVIEW classification. If GENUINE and "
        "auto_register_if_genuine is enabled (default True), the outlet is persisted to PostgreSQL and "
        "photograph uploaded to Railway Object Storage."
    ),
)
def verify_outlet_form(
    name: str = Form(..., description="Outlet name"),
    latitude: float = Form(..., description="Latitude coordinate (-90 to +90)"),
    longitude: float = Form(..., description="Longitude coordinate (-180 to +180)"),
    image: UploadFile = File(..., description="Outlet storefront photograph file"),
    auto_register_if_genuine: bool = Form(
        True,
        description="Automatically persist genuine medical shops to PostgreSQL and store image in Railway S3 bucket",
    ),
    name_top_k: int | None = Form(None),
    image_top_k: int | None = Form(None),
    geo_radius_meters: float | None = Form(None),
    geo_top_k: int | None = Form(None),
    db: Session = Depends(get_db),
    verification_service: VerificationService = Depends(get_verification_service),
) -> VerificationResponse:
    try:
        image_bytes = image.file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded photograph file is empty.",
            )

        filename = image.filename or "storefront.jpg"

        return verification_service.verify_outlet(
            db=db,
            name=name,
            latitude=latitude,
            longitude=longitude,
            image=image_bytes,
            image_filename=filename,
            auto_register_if_genuine=auto_register_if_genuine,
            name_top_k=name_top_k,
            image_top_k=image_top_k,
            geo_radius_meters=geo_radius_meters,
            geo_top_k=geo_top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Error in verification endpoint: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Outlet verification failed: {exc}",
        ) from exc


@router.post(
    "/outlets/register",
    response_model=RegisterOutletResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a verified genuine outlet directly into database and bucket",
    description="Uploads storefront image to Railway S3 Object Storage, computes embeddings, and saves outlet to PostgreSQL.",
)
def register_outlet_direct(
    name: str = Form(..., description="Outlet name"),
    latitude: float = Form(..., description="Latitude coordinate (-90 to +90)"),
    longitude: float = Form(..., description="Longitude coordinate (-180 to +180)"),
    image: UploadFile = File(..., description="Outlet storefront photograph file"),
    db: Session = Depends(get_db),
    verification_service: VerificationService = Depends(get_verification_service),
) -> RegisterOutletResponse:
    try:
        image_bytes = image.file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded photograph file is empty.",
            )

        filename = image.filename or "storefront.jpg"
        new_outlet = verification_service.register_outlet(
            db=db,
            name=name,
            latitude=latitude,
            longitude=longitude,
            image=image_bytes,
            filename=filename,
        )

        return RegisterOutletResponse(
            id=str(new_outlet.id),
            name=new_outlet.name,
            latitude=new_outlet.latitude,
            longitude=new_outlet.longitude,
            image_url=new_outlet.image_url,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Error registering outlet: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Outlet registration failed: {exc}",
        ) from exc


@router.get(
    "/outlets/images/{image_path:path}",
    summary="Fetch outlet image directly from Railway Object Storage",
    description="Streams the raw image from the Railway bucket with caching headers.",
)
def get_outlet_image_proxy(image_path: str):
    try:
        storage = get_storage_service()
        data_bytes, content_type = storage.get_object_bytes(image_path)
        return Response(
            content=data_bytes,
            media_type=content_type,
            headers={
                "Cache-Control": "public, max-age=604800, immutable",
            },
        )
    except Exception as exc:
        logger.error("Failed to retrieve image '%s': %s", image_path, exc)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found in storage bucket.")


@router.post(
    "/verify-json",
    response_model=VerificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify submitted outlet via JSON payload",
    description="Development and testing endpoint accepting base64 image or image URL.",
)
def verify_outlet_json(
    req: VerificationJsonRequest,
    db: Session = Depends(get_db),
    verification_service: VerificationService = Depends(get_verification_service),
) -> VerificationResponse:
    try:
        if req.image_base64:
            image_input: Any = base64.b64decode(req.image_base64)
        elif req.image_url:
            image_input = req.image_url
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'image_base64' or 'image_url' must be provided.",
            )

        return verification_service.verify_outlet(
            db=db,
            name=req.name,
            latitude=req.latitude,
            longitude=req.longitude,
            image=image_input,
            auto_register_if_genuine=req.auto_register_if_genuine,
            name_top_k=req.name_top_k,
            image_top_k=req.image_top_k,
            geo_radius_meters=req.geo_radius_meters,
            geo_top_k=req.geo_top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Error in verification-json endpoint: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Outlet verification failed: {exc}",
        ) from exc
