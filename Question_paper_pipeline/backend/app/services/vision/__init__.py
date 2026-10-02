"""
Vision verification package for the Gecko Med Question Paper Extraction Pipeline.
Provides selective, parallel multimodal verification using Gemini Vision.
"""

from app.services.vision.gemini_client import verify_page_with_gemini
from app.services.vision.correction_validator import validate_and_filter_corrections
from app.services.vision.vision_manager import VisionPipelineManager

__all__ = [
    "verify_page_with_gemini",
    "validate_and_filter_corrections",
    "VisionPipelineManager",
]
