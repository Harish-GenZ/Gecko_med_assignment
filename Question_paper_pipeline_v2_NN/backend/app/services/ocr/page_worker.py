"""
Standalone page worker function for multiprocessing.

Must remain in a top-level importable module so Windows spawn-based
multiprocessing can serialize and deserialize tasks cleanly.
"""

import os
import sys

# Crucial thread safety settings for Windows multiprocessing with PaddlePaddle & OpenBLAS:
# Prevents multi-thread memory aborts, CPU thrashing, and abrupt worker process termination
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_ENABLE_MKLDNN"] = "0"

import time
import json
import logging
from typing import Dict, Any

from app.models.schemas import PageType
from app.services.preprocessing.image_preprocessor import preprocess
from app.services.ocr.ocr_engine import ocr_image, extract_text_native
from app.core.config import JOBS_DIR

# In spawned worker child processes on Windows, ensure logging is configured to console
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

logger = logging.getLogger(__name__)


def _update_page_status(job_id: str, page_index: int, pid: int, status: str, start_time: float) -> None:
    if not job_id:
        return
    try:
        status_file = JOBS_DIR / f"{job_id}_p{page_index}.json"
        temp_file = JOBS_DIR / f"{job_id}_p{page_index}.tmp"
        data = {
            "worker_pid": pid,
            "page_number": page_index + 1,
            "status": status,
            "elapsed_seconds": round(time.time() - start_time, 1),
        }
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
        temp_file.replace(status_file)
    except Exception:
        pass


def process_single_page_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single document page independently in a worker process.
    Fault-tolerant: Catches internal exceptions to prevent crashing the worker process.
    """
    page_index = task_data["page_index"]
    page_type = task_data["page_type"]
    native_text = task_data.get("native_text", "")
    image = task_data.get("image")
    job_id = task_data.get("job_id", "")
    pid = os.getpid()

    start_time = time.time()
    type_name = page_type.value if hasattr(page_type, "value") else str(page_type)
    logger.info(f"[Worker PID: {pid}] ---> STARTING Page {page_index + 1} ({type_name.upper()})")
    _update_page_status(job_id, page_index, pid, f"Starting {type_name.upper()} extraction", start_time)

    try:
        if page_type == PageType.DIGITAL or type_name.lower() == "digital":
            _update_page_status(job_id, page_index, pid, "Extracting native vector text...", start_time)
            ocr_result = extract_text_native(native_text)
        else:
            if image is None:
                raise ValueError(f"Page {page_index + 1} marked as SCANNED but no image provided.")
            logger.info(f"[Worker PID: {pid}] Page {page_index + 1}: Preprocessing image...")
            _update_page_status(job_id, page_index, pid, "Preprocessing image (contrast/resolution)...", start_time)
            preprocessed = preprocess(image)
            logger.info(f"[Worker PID: {pid}] Page {page_index + 1}: Running RapidOCR ONNX inference (with layout noise masking)...")
            _update_page_status(job_id, page_index, pid, "Running RapidOCR ONNX neural text recognition...", start_time)
            ocr_result = ocr_image(preprocessed)

        duration = time.time() - start_time
        logger.info(
            f"[Worker PID: {pid}] <--- FINISHED Page {page_index + 1} in {duration:.1f}s "
            f"({len(ocr_result['text'])} chars extracted, confidence={ocr_result['confidence']:.2f})"
        )
        _update_page_status(job_id, page_index, pid, f"Completed ({len(ocr_result['text'])} chars)", start_time)

        return {
            "page_index": page_index,
            "page_type": page_type,
            "text": ocr_result["text"],
            "ocr_conf": ocr_result["confidence"],
            "ocr_used": ocr_result["ocr_used"],
        }
    except Exception as e:
        duration = time.time() - start_time
        logger.exception(f"[Worker PID: {pid}] Page {page_index + 1} failed after {duration:.1f}s: {e}")
        _update_page_status(job_id, page_index, pid, f"Failed: {str(e)[:30]}", start_time)
        return {
            "page_index": page_index,
            "page_type": page_type,
            "text": native_text if native_text else "",
            "ocr_conf": 0.10,
            "ocr_used": True,
            "error": str(e),
        }



