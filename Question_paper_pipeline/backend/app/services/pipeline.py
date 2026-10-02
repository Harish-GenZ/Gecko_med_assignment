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
import logging
import threading
from pathlib import Path
from typing import List, Optional, Callable

from app.core.config import OCR_MIN_CONFIDENCE
from app.core import job_store
from app.models.schemas import (
    Question, QuestionPaper, ExtractionResult, PageInfo,
    PageType, JobStatus, ReviewStatus, MCQOption, QuestionType
)
from app.services.ingestion.ingestor import ingest_pdf, ingest_images
from app.services.preprocessing.image_preprocessor import preprocess
from app.services.ocr.ocr_engine import ocr_image, extract_text_native
from app.services.layout.boundary_detector import assign_pages_to_papers
from app.services.structure.document_structure import analyze_document_structure
from app.services.extraction.metadata_extractor import extract_metadata
from app.services.segmentation.question_segmenter import segment_questions
from app.services.classification.question_classifier import classify_question
from app.services.validation.validator import validate_and_flag

logger = logging.getLogger(__name__)

# Regex to detect section context for classification inheritance
_SECTION_CONTEXT_RE = re.compile(
    r'(long\s+question|essay|short\s+question|short\s+notes?|'
    r'short\s+answer|very\s+short|multiple\s+choice|mcq|objective)',
    re.I
)


def run_pipeline(file_path: Path, job_id: str) -> None:
    """
    Entry point for background processing.
    Updates job store with progress and final result.
    """
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
        logger.info(f"[{job_id}] Ingested {total_pages} pages from {file_path.name}")
        _update(job_id, progress=10, current_step=f"Processing {total_pages} pages")

        # ------------------------------------------------------------------
        # Step 2: Preprocess + OCR
        # ------------------------------------------------------------------
        pages_info = []
        for i, page in enumerate(pages):
            progress = 10 + int(60 * i / max(total_pages, 1))
            _update(job_id, progress=progress,
                    current_step=f"OCR page {i+1}/{total_pages}")

            if page["page_type"] == PageType.DIGITAL:
                ocr_result = extract_text_native(page["native_text"])
            else:
                preprocessed = preprocess(page["image"])
                ocr_result   = ocr_image(preprocessed)

            page["text"]       = ocr_result["text"]
            page["ocr_conf"]   = ocr_result["confidence"]
            page["ocr_used"]   = ocr_result["ocr_used"]

            pages_info.append(PageInfo(
                page_number = page["page_index"] + 1,
                type        = page["page_type"],
                ocr_used    = page["ocr_used"],
                char_count  = len(page["text"]),
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

            # OCR confidence for paper (average of page confidences)
            avg_ocr_conf = sum(
                p.get("ocr_conf", 0) for p in paper_pages
            ) / max(len(paper_pages), 1)

            distinct_parts = list(dict.fromkeys(q.part for q in questions if q.part))
            if not distinct_parts and doc_structure.sections:
                distinct_parts = doc_structure.sections

            question_papers.append(QuestionPaper(
                paper_index  = paper_idx,
                page_range   = [start_pg, end_pg],
                metadata     = metadata,
                questions    = questions,
                instructions = doc_structure.instructions if doc_structure.instructions else None,
                parts        = distinct_parts if distinct_parts else None,
                raw_text     = full_text[:5000],  # Truncate for storage
            ))

        # ------------------------------------------------------------------
        # Step 8: Validation
        # ------------------------------------------------------------------
        _update(job_id, progress=95, current_step="Validating results")

        result = ExtractionResult(
            document        = file_path.name,
            total_pages     = total_pages,
            question_papers = question_papers,
            processing_notes = [],
        )
        result = validate_and_flag(result)

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
        logger.info(f"[{job_id}] Pipeline complete: {len(question_papers)} papers, "
                    f"{sum(len(p.questions) for p in question_papers)} questions")

    except Exception as e:
        logger.exception(f"[{job_id}] Pipeline failed: {e}")
        _update(
            job_id,
            status       = JobStatus.FAILED,
            current_step = "Failed",
            error        = str(e),
        )


def _update(job_id: str, **kwargs) -> None:
    job_store.update_job(job_id, **kwargs)
