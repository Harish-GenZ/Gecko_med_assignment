"""
Main pipeline orchestrator.

Runs the full extraction pipeline for a single document:
  1. Ingest (detect page types)
  2. Preprocess (deskew, denoise, binarise)
  3. OCR / text extraction
  4. Paper boundary detection
  5. Per-paper: metadata extraction
  6. Per-paper: question segmentation
  7. Per-paper: question classification
  8. Validation
  9. Return ExtractionResult

Designed to run as a background thread.
Progress updates are written to the job store.
"""

import re
import time
import logging
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Callable

from app.core.config import OCR_MIN_CONFIDENCE, OCR_WORKERS
from app.core import job_store
from app.models.schemas import (
    Question, QuestionPaper, ExtractionResult, PageInfo,
    PageType, JobStatus, ReviewStatus, MCQOption, QuestionType,
    PageProcessingState, AnomalySeverity
)
from app.services.ingestion.ingestor import ingest_pdf, ingest_images
from app.services.preprocessing.image_preprocessor import preprocess
from app.services.ocr.ocr_engine import ocr_image, extract_text_native
from app.services.ocr.page_worker import process_single_page_worker
from app.services.layout.boundary_detector import assign_pages_to_papers
from app.services.structure.document_structure import analyze_document_structure
from app.services.extraction.metadata_extractor import extract_metadata
from app.services.segmentation.question_segmenter import segment_questions
from app.services.classification.question_classifier import classify_question
from app.services.validation.validator import validate_and_flag
from app.services.validation.quality_audit import audit_paper, audit_document_structure, audit_ocr
from app.services.vision.vision_manager import VisionPipelineManager

logger = logging.getLogger(__name__)

# Regex to detect section context for classification inheritance
_SECTION_CONTEXT_RE = re.compile(
    r'(long\s+question|essay|short\s+question|short\s+notes?|'
    r'short\s+answer|very\s+short|multiple\s+choice|mcq|objective)',
    re.I
)


def _evaluate_and_trigger_page_vision(
    page: dict,
    idx: int,
    vision_manager: VisionPipelineManager,
    job_id: str,
    total_pages: int = 1,
) -> None:
    """
    Evaluates page-level OCR text and triggers early Vision verification if anomalies detected
    or if the document is a single/short page or camera/scan image where high fidelity is critical.
    Does NOT block: submits task to background thread pool and returns immediately.
    """
    text = page.get("text", "")
    if len(text.strip()) < 30:
        vision_manager.set_page_state(idx, PageProcessingState.AUDIT_COMPLETE)
        return

    try:
        doc_struct = analyze_document_structure(text)
        raw_qs = segment_questions(doc_struct.questions_raw_text or text)
        page_qs = []
        for raw_q in raw_qs:
            q_type, q_conf = classify_question(raw_q, raw_q.get("section"))
            page_qs.append(Question(
                number=raw_q["number"],
                text=raw_q["text"],
                type=q_type,
                part=raw_q.get("part") or raw_q.get("section"),
                heading=raw_q.get("heading"),
                marks=raw_q.get("marks"),
                confidence=round(q_conf, 3),
                status=ReviewStatus.OK
            ))

        _, _, ocr_anomalies = audit_ocr([page])
        struct_conf, struct_anomalies = audit_document_structure(
            instructions=doc_struct.instructions,
            sections=doc_struct.sections,
            header_text=doc_struct.header_text,
            questions=page_qs,
            raw_text=text
        )
        all_anomalies = ocr_anomalies + struct_anomalies

        needs_vision = False
        reasons = []
        for a in all_anomalies:
            if a.severity in (AnomalySeverity.HIGH, AnomalySeverity.CRITICAL):
                needs_vision = True
                reasons.append(a.type)

        if struct_conf < 0.65 and "low_structure_confidence" not in reasons:
            needs_vision = True
            reasons.append("low_structure_confidence")

        # For single/few-page documents or non-digital scanned/photo pages:
        # User requirement: take the extra seconds (up to 15-20s) to extract high-fidelity details with Vision
        is_scanned_or_photo = page.get("page_type") != PageType.DIGITAL
        if vision_manager.enabled:
            if total_pages <= 3:
                needs_vision = True
                reasons.append("short_document_high_fidelity")
            elif is_scanned_or_photo:
                needs_vision = True
                reasons.append("scanned_or_camera_image")

        if needs_vision:
            extracted_meta = extract_metadata(doc_struct.header_text or text[:1000]).model_dump()
            payload = {
                "ocr_text": text,
                "metadata": extracted_meta,
                "instructions": doc_struct.instructions or [],
                "sections": doc_struct.sections or [],
                "questions": [q.model_dump() for q in page_qs],
                "confidence": {
                    "ocr": page.get("ocr_conf"),
                    "structure": struct_conf,
                },
                "verification_reasons": reasons,
            }
            # SUBMIT ASYNC VISION TASK IMMEDIATELY (NON-BLOCKING)
            vision_manager.submit_page_verification(
                page_index=idx,
                page_image=page.get("image"),
                page_payload=payload,
                reasons=reasons,
            )
        else:
            vision_manager.set_page_state(idx, PageProcessingState.AUDIT_COMPLETE)
    except Exception as e:
        logger.warning(f"[{job_id}] Early vision check failed for page {idx + 1}: {e}")
        vision_manager.set_page_state(idx, PageProcessingState.AUDIT_COMPLETE)


def run_pipeline(file_path: Path, job_id: str) -> None:
    """
    Entry point for background processing.
    Updates job store with progress and final result.
    """
    vision_manager = None
    try:
        _update(job_id, status=JobStatus.PROCESSING, progress=5, current_step="Ingesting document")

        # ------------------------------------------------------------------
        # Step 1: Ingest
        # ------------------------------------------------------------------
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            pages = ingest_pdf(file_path)
        elif suffix in (".jpg", ".jpeg", ".png"):
            pages = ingest_images([file_path])
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        total_pages = len(pages)
        pipeline_start_time = time.time()
        logger.info(f"\n{'='*70}\n[JOB START: {job_id}] File: '{file_path.name}' | Total Pages: {total_pages}\n{'='*70}")
        _update(job_id, progress=10, current_step=f"Processing {total_pages} pages")

        # Initialize Vision Manager for asynchronous non-blocking verification
        vision_manager = VisionPipelineManager(job_id=job_id)

        # ------------------------------------------------------------------
        # Step 2: Preprocess + OCR (Parallel page-level processing)
        # ------------------------------------------------------------------
        max_workers = min(OCR_WORKERS, total_pages)

        if total_pages <= 1 or max_workers <= 1:
            # Single-page: run sequentially without process pool spawn overhead
            results = []
            for i, page in enumerate(pages):
                _update(job_id, progress=10, current_step="OCR page 1/1")
                task_payload = {
                    "job_id":      job_id,
                    "page_index":  page["page_index"],
                    "page_type":   page["page_type"],
                    "native_text": page["native_text"],
                    "image":       page["image"] if page["page_type"] != PageType.DIGITAL else None,
                }
                res = process_single_page_worker(task_payload)
                results.append(res)
                idx = res["page_index"]
                page["text"] = res["text"]
                page["ocr_conf"] = res["ocr_conf"]
                page["ocr_used"] = res["ocr_used"]
                if res.get("image") is not None:
                    page["image"] = res["image"]
                vision_manager.set_page_state(idx, PageProcessingState.OCR_COMPLETE)
                vision_manager.record_timestamp(idx, "ocr_completed")
                _evaluate_and_trigger_page_vision(page, idx, vision_manager, job_id, total_pages=total_pages)
        else:
            # Multi-page: distribute independent pages across ProcessPoolExecutor workers
            logger.info(f"[{job_id}] Distributing {total_pages} pages across {max_workers} worker processes (OCR_WORKERS={OCR_WORKERS})")
            results = []
            completed_count = 0
            processed_indices = set()

            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_to_page = {}
                    page_iter = iter(pages)

                    # Prime the pool with an initial bounded batch (prevents IPC queue flooding on 48+ page PDFs)
                    batch_size = min(max_workers * 2, total_pages)
                    for _ in range(batch_size):
                        try:
                            p = next(page_iter)
                            task_payload = {
                                "job_id":      job_id,
                                "page_index":  p["page_index"],
                                "page_type":   p["page_type"],
                                "native_text": p["native_text"],
                                "image":       p["image"] if p["page_type"] != PageType.DIGITAL else None,
                            }
                            fut = executor.submit(process_single_page_worker, task_payload)
                            future_to_page[fut] = p
                            logger.info(f"[{job_id}] Submitted Page {p['page_index'] + 1}/{total_pages} to process pool")
                        except StopIteration:
                            break

                    while future_to_page:
                        if job_store.is_job_cancelled(job_id):
                            logger.info(f"[{job_id}] User cancelled job during worker execution. Cancelling remaining tasks.")
                            for fut in future_to_page:
                                fut.cancel()
                            return

                        # Wait for the next completed future
                        done_fut = next(as_completed(list(future_to_page.keys())))
                        p = future_to_page.pop(done_fut)
                        try:
                            res = done_fut.result()
                        except Exception as page_exc:
                            logger.error(f"[{job_id}] Process worker error on Page {p['page_index'] + 1}: {page_exc}")
                            res = {
                                "page_index": p["page_index"],
                                "page_type":  p["page_type"],
                                "text":       p.get("native_text", ""),
                                "ocr_conf":   0.10,
                                "ocr_used":   True,
                            }

                        results.append(res)
                        completed_count += 1
                        idx = res["page_index"]
                        processed_indices.add(idx)

                        page = pages[idx]
                        page["text"] = res["text"]
                        page["ocr_conf"] = res["ocr_conf"]
                        page["ocr_used"] = res["ocr_used"]
                        if res.get("image") is not None:
                            page["image"] = res["image"]
                        vision_manager.set_page_state(idx, PageProcessingState.OCR_COMPLETE)
                        vision_manager.record_timestamp(idx, "ocr_completed")

                        # Trigger Vision verification asynchronously immediately
                        _evaluate_and_trigger_page_vision(page, idx, vision_manager, job_id, total_pages=total_pages)

                        progress = 10 + int(60 * completed_count / max(total_pages, 1))
                        _update(job_id, progress=progress,
                                current_step=f"OCR {completed_count}/{total_pages} pages completed")
                        logger.info(f"[{job_id}] OCR completed for page {idx + 1}/{total_pages} ({len(res['text'])} chars)")

                        # Feed next page into the pool as soon as one completes
                        try:
                            next_p = next(page_iter)
                            next_payload = {
                                "job_id":      job_id,
                                "page_index":  next_p["page_index"],
                                "page_type":   next_p["page_type"],
                                "native_text": next_p["native_text"],
                                "image":       next_p["image"] if next_p["page_type"] != PageType.DIGITAL else None,
                            }
                            next_fut = executor.submit(process_single_page_worker, next_payload)
                            future_to_page[next_fut] = next_p
                            logger.info(f"[{job_id}] Submitted Page {next_p['page_index'] + 1}/{total_pages} to process pool")
                        except StopIteration:
                            pass

            except Exception as pool_err:
                logger.error(f"[{job_id}] ProcessPool exception encountered: {pool_err}. Falling back to sequential recovery for remaining pages.")
                for p in pages:
                    idx = p["page_index"]
                    if idx not in processed_indices:
                        fallback_payload = {
                            "job_id":      job_id,
                            "page_index":  idx,
                            "page_type":   p["page_type"],
                            "native_text": p["native_text"],
                            "image":       p["image"] if p["page_type"] != PageType.DIGITAL else None,
                        }
                        try:
                            res = process_single_page_worker(fallback_payload)
                        except Exception as e:
                            logger.error(f"[{job_id}] Sequential recovery failed for Page {idx + 1}: {e}")
                            res = {
                                "page_index": idx,
                                "page_type":  p["page_type"],
                                "text":       p.get("native_text", ""),
                                "ocr_conf":   0.10,
                                "ocr_used":   True,
                            }
                        results.append(res)
                        completed_count += 1
                        processed_indices.add(idx)

                        page = pages[idx]
                        page["text"] = res["text"]
                        page["ocr_conf"] = res["ocr_conf"]
                        page["ocr_used"] = res["ocr_used"]
                        if res.get("image") is not None:
                            page["image"] = res["image"]
                        vision_manager.set_page_state(idx, PageProcessingState.OCR_COMPLETE)
                        vision_manager.record_timestamp(idx, "ocr_completed")
                        _evaluate_and_trigger_page_vision(page, idx, vision_manager, job_id, total_pages=total_pages)

                        progress = 10 + int(60 * completed_count / max(total_pages, 1))
                        _update(job_id, progress=progress,
                                current_step=f"OCR {completed_count}/{total_pages} pages completed")

        # Restore original page order (crucial for boundary detection & downstream pipeline)
        results.sort(key=lambda r: r["page_index"])

        page_states = {p["page_index"]: vision_manager.get_page_state(p["page_index"]) for p in pages}
        pages_info = []
        for res in results:
            idx = res["page_index"]
            pages_info.append(PageInfo(
                page_number      = idx + 1,
                type             = res["page_type"],
                ocr_used         = res["ocr_used"],
                char_count       = len(res["text"]),
                processing_state = page_states.get(idx, PageProcessingState.FINALIZED),
            ))

        _update(job_id, progress=72, current_step="Detecting paper boundaries",
                pages_info=[p.model_dump() for p in pages_info])

        # ------------------------------------------------------------------
        # Step 3: Paper boundary detection
        # ------------------------------------------------------------------
        paper_groups = assign_pages_to_papers(pages)
        logger.info(f"[{job_id}] Detected {len(paper_groups)} question paper(s)")
        _update(job_id, progress=76, current_step=f"Extracting {len(paper_groups)} paper(s)")

        # ------------------------------------------------------------------
        # Step 4–7: Per-paper processing
        # ------------------------------------------------------------------
        question_papers = []
        for paper_idx, paper_pages in enumerate(paper_groups):
            _update(job_id, progress=76 + int(18 * paper_idx / max(len(paper_groups), 1)),
                    current_step=f"Processing paper {paper_idx + 1}/{len(paper_groups)}")

            # Combine all page text for this paper
            full_text = "\n".join(p["text"] for p in paper_pages)

            # Page range (1-indexed)
            start_pg = paper_pages[0]["page_index"] + 1
            end_pg   = paper_pages[-1]["page_index"] + 1

            # --------------------------------------------------------------
            # Step 4: Document Structure Stage
            # Separates HEADER/METADATA, INSTRUCTIONS, SECTION HEADINGS,
            # and ACTUAL QUESTION BLOCKS before any extraction.
            # --------------------------------------------------------------
            doc_structure = analyze_document_structure(full_text)
            logger.info(f"[{job_id}] Paper {paper_idx + 1}: Document structure parsed — "
                        f"{len(doc_structure.instructions)} instruction item(s), "
                        f"{len(doc_structure.sections)} section(s)")

            # Step 5: Metadata extraction (strictly from header region)
            header_source = doc_structure.header_text or "\n".join(p["text"] for p in paper_pages[:2])
            metadata = extract_metadata(header_source)
            logger.info(f"[{job_id}] Paper {paper_idx + 1}: Metadata — "
                        f"Univ: '{metadata.university or 'N/A'}', "
                        f"Code: '{metadata.session_code or 'N/A'}', "
                        f"Subject: '{metadata.subject or 'N/A'}'")

            # Step 6: Question segmentation (strictly on genuine question blocks)
            questions_source = doc_structure.questions_raw_text or full_text
            raw_questions = segment_questions(questions_source)
            logger.info(f"[{job_id}] Paper {paper_idx + 1}: Segmented {len(raw_questions)} question block(s)")

            # Step 7: Classification (only genuine question blocks enter classifier)
            questions = []
            for raw_q in raw_questions:
                q_type, confidence = classify_question(raw_q, raw_q.get("section"))

                # Parse MCQ options ONLY if question type is MCQ
                mcq_opts = None
                if q_type == QuestionType.MCQ and raw_q.get("options"):
                    mcq_opts = [
                        MCQOption(label=o["label"], text=o["text"])
                        for o in raw_q.get("options", [])
                    ]

                status = (ReviewStatus.NEEDS_REVIEW
                          if confidence < 0.60
                          else ReviewStatus.OK)

                part_val = raw_q.get("part") or raw_q.get("section")
                questions.append(Question(
                    number     = raw_q["number"],
                    text       = raw_q["text"],
                    type       = q_type,
                    part       = part_val,
                    heading    = raw_q.get("heading"),
                    options    = mcq_opts,
                    marks      = raw_q.get("marks"),
                    confidence = round(confidence, 3),
                    status     = status,
                ))

            distinct_parts = list(dict.fromkeys(q.part for q in questions if q.part))
            if not distinct_parts and doc_structure.sections:
                distinct_parts = doc_structure.sections

            qp = QuestionPaper(
                paper_index  = paper_idx,
                page_range   = [start_pg, end_pg],
                metadata     = metadata,
                questions    = questions,
                instructions = doc_structure.instructions if doc_structure.instructions else None,
                parts        = distinct_parts if distinct_parts else None,
                raw_text     = full_text[:5000],  # Truncate for storage
            )

            # Master multi-dimensional quality audit
            qp = audit_paper(qp, paper_pages, doc_structure)

            # Check if whole paper audit detected an anomaly on pages not yet submitted to Vision
            should_verify_paper = (
                qp.needs_visual_verification
                or total_pages <= 3
                or any(p.get("page_type") != PageType.DIGITAL for p in paper_pages)
            )
            if should_verify_paper and vision_manager.enabled:
                for p in paper_pages:
                    p_idx = p["page_index"]
                    if p_idx not in vision_manager._futures:
                        payload = {
                            "ocr_text": p.get("text", ""),
                            "metadata": qp.metadata.model_dump() if qp.metadata else {},
                            "instructions": qp.instructions or [],
                            "sections": qp.parts or [],
                            "questions": [q.model_dump() for q in qp.questions],
                            "confidence": {
                                "ocr": p.get("ocr_conf"),
                                "structure": qp.confidence.structure_confidence if qp.confidence else 0.5,
                                "overall": qp.confidence.overall_confidence if qp.confidence else 0.5,
                            },
                            "verification_reasons": qp.visual_verification_reasons or ["paper_level_verification"],
                        }
                        vision_manager.submit_page_verification(
                            page_index=p_idx,
                            page_image=p.get("image"),
                            page_payload=payload,
                            reasons=qp.visual_verification_reasons or ["paper_level_verification"],
                        )

            # Await any running Vision tasks (returns instantly if already completed during OCR)
            _update(job_id, progress=92, current_step=f"Verifying structure for Paper {paper_idx + 1}")
            vision_manager.await_all_completed(timeout=45.0)

            # Apply corrections & revalidate deterministically
            qp, applied = vision_manager.apply_corrections_and_revalidate(qp, paper_pages, doc_structure)
            question_papers.append(qp)

        # ------------------------------------------------------------------
        # Step 8: Validation
        # ------------------------------------------------------------------
        page_states = {p["page_index"]: vision_manager.get_page_state(p["page_index"]) for p in pages}
        pages_info = []
        for res in results:
            idx = res["page_index"]
            pages_info.append(PageInfo(
                page_number      = idx + 1,
                type             = res["page_type"],
                ocr_used         = res["ocr_used"],
                char_count       = len(res["text"]),
                processing_state = page_states.get(idx, PageProcessingState.FINALIZED),
            ))

        _update(job_id, progress=95, current_step="Validating results",
                pages_info=[p.model_dump() for p in pages_info])

        result = ExtractionResult(
            document        = file_path.name,
            total_pages     = total_pages,
            question_papers = question_papers,
            processing_notes = [],
        )
        result = validate_and_flag(
            result,
            pages_list=pages,
            page_states=page_states,
            vision_results=vision_manager._results,
        )

        # ------------------------------------------------------------------
        # Complete
        # ------------------------------------------------------------------
        _update(
            job_id,
            status       = JobStatus.COMPLETED,
            progress     = 100,
            current_step = "Completed",
            result       = result.model_dump(),
        )
        total_duration = time.time() - pipeline_start_time
        total_questions = sum(len(p.questions) for p in question_papers)
        logger.info(
            f"\n{'='*70}\n[JOB COMPLETE: {job_id}]\n"
            f"  Status: SUCCESS\n"
            f"  Question Papers Detected: {len(question_papers)}\n"
            f"  Total Questions Extracted: {total_questions}\n"
            f"  Total Execution Time: {total_duration:.2f}s ({total_duration/max(total_pages,1):.2f}s per page)\n"
            f"{'='*70}\n"
        )
        return result

    except Exception as e:
        if job_store.is_job_cancelled(job_id):
            logger.info(f"[{job_id}] Pipeline aborted gracefully following cancellation.")
            return
        logger.error(f"\n{'!'*70}\n[JOB FAILED: {job_id}]\n  Error: {e}\n{'!'*70}")
        logger.exception(f"[{job_id}] Traceback:")
        _update(
            job_id,
            status       = JobStatus.FAILED,
            current_step = "Failed",
            error        = str(e),
        )
    finally:
        if 'vision_manager' in locals() and vision_manager is not None:
            vision_manager.shutdown()
        _cleanup_worker_status_files(job_id)


def _cleanup_worker_status_files(job_id: str) -> None:
    try:
        from app.core.config import JOBS_DIR
        for p in JOBS_DIR.glob(f"{job_id}_p*.*"):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
    except Exception:
        pass


def _update(job_id: str, **kwargs) -> None:
    if job_store.is_job_cancelled(job_id):
        return
    job_store.update_job(job_id, **kwargs)
