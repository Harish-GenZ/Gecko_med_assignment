"""
FastAPI router for upload and job management endpoints.
"""

import logging
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.core import job_store
from app.core.config import UPLOADS_DIR, MAX_FILE_SIZE_MB
from app.models.schemas import JobCreateResponse, JobStatusResponse, JobStatus
from app.services.pipeline import run_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["pipeline"])

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}


@router.post("/upload", response_model=JobCreateResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or image file for processing.
    Returns a job_id that can be polled for status.
    """
    logger.info(f"Incoming upload request for: '{file.filename}'")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}"
        )

    # Save uploaded file
    safe_name = f"{uuid.uuid4()}{suffix}"
    save_path = UPLOADS_DIR / safe_name

    content = await file.read()

    size_mb = len(content) / 1024 / 1024
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max {MAX_FILE_SIZE_MB} MB."
        )

    with open(save_path, "wb") as f:
        f.write(content)

    # Create job
    job_id = job_store.create_job(file.filename)

    # Launch background processing thread
    t = threading.Thread(
        target=run_pipeline,
        args=(save_path, job_id),
        daemon=True,
        name=f"pipeline-{job_id[:8]}",
    )
    t.start()
    logger.info(f"Created job {job_id} for '{file.filename}'. Dispatched pipeline thread.")

    return JobCreateResponse(
        job_id   = job_id,
        status   = JobStatus.QUEUED,
        filename = file.filename,
        message  = "Document uploaded successfully. Processing started.",
    )


@router.get("/job/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """
    Poll extraction job status and result.
    """
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return job


@router.get("/jobs")
def list_jobs():
    """List all known jobs (for debugging/admin)."""
    from app.core.job_store import _jobs
    return [
        {"job_id": jid, "status": v["status"], "filename": v["filename"]}
        for jid, v in _jobs.items()
    ]


@router.post("/job/{job_id}/cancel")
def cancel_job(job_id: str):
    """
    Cancel an active processing job.
    """
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    job_store.cancel_job(job_id)
    logger.info(f"Job {job_id} cancelled via client request.")
    return {"success": True, "message": f"Job {job_id} cancelled."}


@router.get("/health")
def health():
    return {"status": "ok"}
