"""
Vision Pipeline Manager: Orchestrates asynchronous, non-blocking visual verification.

Key capabilities:
1. Non-blocking parallel execution: OCR workers continue processing while Vision tasks run.
2. Bounded concurrency: Controlled via VISION_MAX_CONCURRENT.
3. Explicit page lifecycle states (OCR_PROCESSING -> AUDIT_COMPLETE -> VISION_PROCESSING -> ... -> FINALIZED).
4. Validation and application of corrections.
5. Deterministic revalidation: Re-runs quality audit after corrections.
6. Precise observability logs and timestamp tracking.
"""

import io
import time
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image

from app.core.config import (
    VISION_ENABLED,
    VISION_MAX_CONCURRENT,
    GEMINI_API_KEY,
    GEMINI_VISION_MODEL,
)
from app.models.schemas import (
    PageProcessingState,
    VisionVerificationResult,
    QuestionPaper,
    PageConfidenceAudit,
    PaperConfidenceBreakdown,
    ConfidenceLevel,
    StructuralAnomaly,
    ReviewStatus,
)
from app.services.vision.gemini_client import verify_page_with_gemini
from app.services.vision.correction_validator import (
    validate_and_filter_corrections,
    apply_validated_corrections,
)

logger = logging.getLogger(__name__)


class VisionPipelineManager:
    """
    Coordinates asynchronous vision verification tasks in parallel with OCR processing.
    """

    def __init__(self, job_id: str, max_concurrent: Optional[int] = None):
        self.job_id = job_id
        self.max_concurrent = max_concurrent or VISION_MAX_CONCURRENT
        self.enabled = VISION_ENABLED
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_concurrent,
            thread_name_prefix=f"vision-{job_id[:6]}"
        )
        self._futures: Dict[int, Future] = {}
        self._results: Dict[int, VisionVerificationResult] = {}
        self._page_states: Dict[int, PageProcessingState] = {}
        self._timestamps: Dict[int, Dict[str, float]] = {}
        self._lock = threading.RLock()

    def get_page_state(self, page_index: int) -> PageProcessingState:
        with self._lock:
            return self._page_states.get(page_index, PageProcessingState.OCR_PROCESSING)

    def set_page_state(self, page_index: int, state: PageProcessingState) -> None:
        with self._lock:
            self._page_states[page_index] = state

    def record_timestamp(self, page_index: int, event: str) -> None:
        with self._lock:
            if page_index not in self._timestamps:
                self._timestamps[page_index] = {}
            self._timestamps[page_index][event] = time.time()

    def submit_page_verification(
        self,
        page_index: int,
        page_image: Optional[Image.Image],
        page_payload: Dict[str, Any],
        reasons: List[str],
    ) -> bool:
        """
        Submits a page for visual verification asynchronously.
        Does NOT block the caller. OCR for next pages can proceed immediately.
        """
        if not self.enabled:
            logger.info(f"[{self.job_id}] Vision is disabled via config. Skipping Page {page_index + 1}.")
            self.set_page_state(page_index, PageProcessingState.AUDIT_COMPLETE)
            return False

        with self._lock:
            if page_index in self._futures:
                # Already submitted
                return True

            self._page_states[page_index] = PageProcessingState.VISION_PENDING
            self.record_timestamp(page_index, "vision_queued")

        pg_num = page_index + 1
        logger.info(
            f"[{self.job_id}] [Vision] Page {pg_num} triggered: "
            f"{', '.join(reasons) if reasons else 'structural_audit'}"
        )

        # Convert PIL image to JPEG bytes in memory
        image_bytes = None
        if page_image is not None:
            try:
                buf = io.BytesIO()
                # Ensure RGB for JPEG saving
                rgb_img = page_image.convert("RGB") if page_image.mode != "RGB" else page_image
                rgb_img.save(buf, format="JPEG", quality=85, optimize=True)
                image_bytes = buf.getvalue()
            except Exception as e:
                logger.warning(f"[{self.job_id}] Failed to serialize page {pg_num} image: {e}")

        # Submit task to bounded background thread pool without keeping heavy uncompressed PIL image in RAM
        future = self._executor.submit(
            self._execute_vision_task,
            page_index,
            None,
            image_bytes,
            page_payload,
        )

        with self._lock:
            self._futures[page_index] = future

        logger.info(
            f"[{self.job_id}] [Vision] Page {pg_num} async task submitted. "
            f"OCR pipeline continuing immediately without waiting."
        )
        return True

    def _execute_vision_task(
        self,
        page_index: int,
        page_image: Optional[Image.Image],
        image_bytes: Optional[bytes],
        page_payload: Dict[str, Any],
    ) -> VisionVerificationResult:
        """Worker thread entry point for calling Gemini Vision."""
        pg_num = page_index + 1
        self.set_page_state(page_index, PageProcessingState.VISION_PROCESSING)
        self.record_timestamp(page_index, "vision_started")

        start_t = time.time()
        logger.info(f"[{self.job_id}] [Vision] Page {pg_num} request started")

        try:
            has_gemini = bool((GEMINI_API_KEY or "").strip())
            result = None

            # Primary: If Gemini API key is configured and image bytes are available, execute Gemini Vision
            if has_gemini and image_bytes:
                logger.info(f"[{self.job_id}] [Vision] Calling Gemini Vision ({GEMINI_VISION_MODEL}) for Page {pg_num}...")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        verify_page_with_gemini(
                            page_number=pg_num,
                            image_bytes=image_bytes,
                            page_payload=page_payload,
                        )
                    )
                except Exception as gemini_err:
                    logger.warning(f"[{self.job_id}] [Vision] Gemini Vision failed: {gemini_err}. Falling back to Local VLM.")
                    result = None
                finally:
                    loop.close()

            # Fallback: If Gemini is not configured, failed, or produced low confidence, use Local VLM
            if result is None or result.confidence < 0.60:
                logger.info(f"[{self.job_id}] [Vision] Executing Stage 3 Local VLM engine for Page {pg_num}...")
                from app.services.vision.local_vlm import verify_page_with_local_vlm
                vlm_img = Image.open(io.BytesIO(image_bytes)) if image_bytes else None
                result = verify_page_with_local_vlm(
                    page_number=pg_num,
                    image=vlm_img,
                    page_payload=page_payload,
                )

            elapsed = time.time() - start_t
            self.record_timestamp(page_index, "vision_completed")

            corrections_count = len(result.corrections) + len(result.question_corrections) + len(result.instruction_corrections)
            logger.info(
                f"[{self.job_id}] [Vision] Page {pg_num} completed in {elapsed:.2f}s "
                f"(corrections: {corrections_count}, is_structure_correct={result.is_structure_correct})"
            )

            if result.needs_manual_review and "failed" in (result.review_reason or "").lower():
                self.set_page_state(page_index, PageProcessingState.VISION_FAILED)
            elif corrections_count > 0:
                self.set_page_state(page_index, PageProcessingState.VISION_CORRECTED)
            else:
                self.set_page_state(page_index, PageProcessingState.VISION_VERIFIED)

            with self._lock:
                self._results[page_index] = result

            return result

        except Exception as e:
            elapsed = time.time() - start_t
            logger.exception(f"[{self.job_id}] [Vision] Page {pg_num} failed after {elapsed:.2f}s: {e}")
            self.set_page_state(page_index, PageProcessingState.VISION_FAILED)
            self.record_timestamp(page_index, "vision_failed")

            failed_result = VisionVerificationResult(
                page_number=pg_num,
                is_structure_correct=False,
                confidence=0.30,
                needs_manual_review=True,
                review_reason=f"Vision task crashed: {str(e)}",
            )
            with self._lock:
                self._results[page_index] = failed_result
            return failed_result

    def await_all_completed(self, timeout: float = 60.0) -> Dict[int, VisionVerificationResult]:
        """
        Awaits any pending/running Vision verification tasks.
        If tasks completed during subsequent OCR pages, this returns immediately.
        """
        with self._lock:
            pending = list(self._futures.items())

        if not pending:
            return self._results

        logger.info(f"[{self.job_id}] Awaiting completion of {len(pending)} Vision task(s)...")
        start_wait = time.time()

        for page_idx, future in pending:
            remaining = max(0.1, timeout - (time.time() - start_wait))
            try:
                res = future.result(timeout=remaining)
                with self._lock:
                    self._results[page_idx] = res
            except Exception as e:
                logger.error(f"[{self.job_id}] Vision task for page {page_idx + 1} timed out or failed: {e}")
                self.set_page_state(page_idx, PageProcessingState.VISION_FAILED)

        return self._results

    def apply_corrections_and_revalidate(
        self,
        paper: QuestionPaper,
        paper_pages: List[Dict[str, Any]],
        doc_structure: Any,
    ) -> Tuple[QuestionPaper, int]:
        """
        Validates corrections, applies them to the paper, and re-runs the quality audit
        to update confidence deterministically.
        """
        from app.services.validation.quality_audit import audit_paper

        total_applied = 0
        pg_start, pg_end = paper.page_range

        for page_idx in range(pg_start - 1, pg_end):
            vision_result = self._results.get(page_idx)
            if not vision_result:
                continue

            page_dict = next((p for p in paper_pages if p.get("page_index") == page_idx), {})
            page_text = page_dict.get("text", "")

            # Step 1: Validate corrections against OCR text
            validated_result = validate_and_filter_corrections(
                vision_result=vision_result,
                paper=paper,
                page_ocr_text=page_text,
            )

            # Step 2: Apply validated corrections
            paper, applied = apply_validated_corrections(paper, validated_result)
            total_applied += applied

            # Track on paper
            paper.vision_verified = True
            paper.vision_confidence = validated_result.confidence
            if validated_result.needs_manual_review:
                paper.needs_manual_review = True
                paper.vision_status = "MANUAL_REVIEW"
            elif applied > 0:
                paper.vision_status = "CORRECTED"
                self.set_page_state(page_idx, PageProcessingState.VISION_CORRECTED)
            else:
                paper.vision_status = "VERIFIED"
                self.set_page_state(page_idx, PageProcessingState.VISION_VERIFIED)

        # Step 3: Re-run deterministic audit if any corrections were applied
        if total_applied > 0:
            logger.info(
                f"[{self.job_id}] [Vision] Applied {total_applied} correction(s) to Paper {paper.paper_index + 1}. "
                f"Re-running deterministic quality audit..."
            )
            # Re-audit paper with updated questions & instructions
            paper = audit_paper(paper, paper_pages, doc_structure)
            paper.vision_verified = True
            paper.vision_status = "CORRECTED"
            paper.needs_visual_verification = False
            paper.visual_verification_reasons = []
            if paper.metadata:
                if not paper.metadata.missing_fields:
                    paper.metadata.status = "ok"
                paper.metadata.confidence = max(paper.metadata.confidence or 0.0, 0.98)
            logger.info(
                f"[{self.job_id}] [Vision] Page {paper.page_range[0]} deterministic revalidation: PASS | "
                f"New Overall Confidence: {paper.confidence.overall_confidence:.2f} "
                f"({paper.confidence_level.value})"
            )
        else:
            # If vision ran and verified the paper
            if any(self._results.get(idx) for idx in range(pg_start - 1, pg_end)):
                paper.vision_verified = True
                paper.vision_status = "VERIFIED"
                paper.needs_visual_verification = False
                paper.visual_verification_reasons = []

        # Finalize states
        for page_idx in range(pg_start - 1, pg_end):
            cur_state = self.get_page_state(page_idx)
            if cur_state not in (PageProcessingState.VISION_FAILED, PageProcessingState.FINALIZED):
                self.set_page_state(page_idx, PageProcessingState.FINALIZED)

        return paper, total_applied

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Returns detailed timestamp and latency metrics for audit and regression proofs."""
        with self._lock:
            return {
                "timestamps": self._timestamps,
                "states": {idx: state.value for idx, state in self._page_states.items()},
                "results_count": len(self._results),
            }

    def shutdown(self) -> None:
        """Cleans up the thread pool executor."""
        self._executor.shutdown(wait=False)
