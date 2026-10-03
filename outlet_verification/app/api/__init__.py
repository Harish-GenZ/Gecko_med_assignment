"""
API routes for Outlet Verification System.
"""

from app.api.candidates import router as candidates_router
from app.api.health import router as health_router
from app.api.verification import router as verification_router

__all__ = ["health_router", "candidates_router", "verification_router"]
