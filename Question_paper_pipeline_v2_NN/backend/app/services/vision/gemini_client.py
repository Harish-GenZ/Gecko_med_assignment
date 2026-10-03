"""
Gemini Vision client for multimodal quality verification of question papers.

Sends page image + parsed structure payload to Gemini Vision.
Returns structured corrections and verification status.
Features:
- Configurable model and API key via environment variables (.env)
- Strict JSON output enforcement
- Bounded retries with exponential backoff
- Timeout protection
- Memory-efficient image handling
"""

import os
import json
import time
import base64
import asyncio
import logging
from typing import Optional, Dict, Any, Callable

import httpx

from app.core.config import (
    GEMINI_API_KEY,
    GEMINI_VISION_MODEL,
    VISION_MAX_RETRIES,
    VISION_TIMEOUT_SECONDS,
)
from app.models.schemas import (
    VisionVerificationResult,
    VisionCorrectionItem,
    VisionQuestionCorrection,
    VisionInstructionCorrection,
    VisionMetadataCorrection,
)

logger = logging.getLogger(__name__)

# Optional mock hook for automated unit tests & offline regression tests
_MOCK_VISION_HANDLER: Optional[Callable[[int, Dict[str, Any]], VisionVerificationResult]] = None


def set_mock_vision_handler(handler: Optional[Callable[[int, Dict[str, Any]], VisionVerificationResult]]) -> None:
    """Set or clear a custom mock vision handler for testing."""
    global _MOCK_VISION_HANDLER
    _MOCK_VISION_HANDLER = handler


SYSTEM_PROMPT = """You are an expert medical examination paper quality auditor and layout verification assistant.
Your task is to inspect the uploaded question paper page image against the machine-extracted text, metadata, instructions, and questions.

CRITICAL INSTRUCTIONS:
1. STRICT SOURCE OF TRUTH — PRINTED / TYPESET DETAILS ONLY:
   - Use ONLY officially printed/typeset text as the source of truth.
   - Question papers frequently have student handwritten notes, pen or pencil scribbles, rough calculations, ticks, crosses, circled MCQ choices, underlines, student names, roll numbers, or margin doodles.
   - You MUST COMPLETELY IGNORE all handwritten annotations, pen/pencil markings, and scribbles.
   - NEVER extract, transcribe, or treat handwritten markings as questions, options, headings, instructions, or metadata.
   - For MCQs, transcribe only the printed options and text; do NOT choose or modify options based on student circles or ticks.

2. EXAM METADATA INSPECTION & CORRECTION:
   - If this page is the first page of an exam paper or contains a header/masthead, inspect the printed details at the top:
     * university: Printed university/institution name (e.g. "Kaloji Narayana Rao University of Health Sciences", "Pt. B.D. Sharma University of Health Sciences", "Bareilly International University").
     * subject: Printed subject/course name (e.g. "FORENSIC MEDICINE", "PATHOLOGY", "PHARMACOLOGY", "BIOCHEMISTRY").
     * exam_year & exam_month: Printed examination session date/month/year (e.g. "NOVEMBER, 2023", "JANUARY 2021", "September 2026").
     * session_code / paper_code: Printed paper code, QP code, or ANQP code (e.g. "MB2019101", "11016", "1725N", "MBY2P5").
     * max_marks: Printed maximum marks / total marks (e.g. "100", "80", "50", "40").
     * duration: Printed time allowed / duration (e.g. "3 Hours", "2 Hours 30 Minutes").
   - Compare the printed masthead with the provided "Extracted Exam Metadata".
   - If any metadata field is missing, incomplete, truncated, or incorrect, output "metadata_corrections" containing the exact printed values.

3. STRUCTURE & QUESTION LEAKAGE:
   - Examine whether text classified as 'instructions' actually contains real questions (e.g., questions with marks like '(8 marks)', numbered questions, or essay/short-note prompts).
   - If real questions were trapped inside instructions, output specific 'corrections' to move them from 'instructions' to 'questions'.
   - Verify question boundaries, numbering, and section assignments (e.g. PART-A, PART-B).

4. ANTI-HALLUCINATION & FIDELITY:
   - DO NOT rewrite or hallucinate content. Only reference or extract text that is visibly printed on the page.
   - If text is illegible or ambiguous, set needs_manual_review to true with a clear reason, rather than guessing.
   - Return ONLY valid JSON adhering strictly to the required schema.

Required JSON Structure:
{
  "page_number": <int>,
  "is_structure_correct": <bool>,
  "confidence": <float 0.0-1.0>,
  "corrections": [
    {
      "type": "move_text",
      "source": "instructions",
      "target": "questions",
      "question_number": "1",
      "section": "PART-A",
      "reason": "Real question with marks was incorrectly placed into instructions"
    }
  ],
  "question_corrections": [
    {
      "question_number": "1",
      "action": "ADD",
      "section": "PART-A",
      "text": "<exact printed text of question from image/OCR>",
      "heading": "<optional prompt heading>",
      "marks": "<marks e.g. 8>",
      "confidence": 0.95
    }
  ],
  "instruction_corrections": [
    {
      "action": "REMOVE",
      "text": "<text to remove from instructions>",
      "reason": "Moved to questions"
    }
  ],
  "metadata_corrections": [
    {
      "field": "university",
      "value": "<exact printed university name>",
      "confidence": 0.95
    },
    {
      "field": "subject",
      "value": "<exact printed subject name>",
      "confidence": 0.95
    },
    {
      "field": "session_code",
      "value": "<exact printed paper or QP code>",
      "confidence": 0.95
    },
    {
      "field": "exam_year",
      "value": "<printed year e.g. 2023>",
      "confidence": 0.95
    },
    {
      "field": "exam_month",
      "value": "<printed month e.g. November>",
      "confidence": 0.95
    },
    {
      "field": "max_marks",
      "value": "<printed marks e.g. 100>",
      "confidence": 0.95
    },
    {
      "field": "duration",
      "value": "<printed duration e.g. 3 Hours>",
      "confidence": 0.95
    }
  ],
  "needs_manual_review": <bool>,
  "review_reason": <string or null>,
  "raw_explanation": "<brief explanation of findings>"
}
"""


def _build_user_message(page_number: int, page_payload: Dict[str, Any]) -> str:
    """Builds the textual part of the prompt containing the audit signals, metadata, and OCR text."""
    ocr_text = page_payload.get("ocr_text", "")
    metadata = page_payload.get("metadata", {})
    instructions = page_payload.get("instructions", [])
    sections = page_payload.get("sections", [])
    questions = page_payload.get("questions", [])
    confidence = page_payload.get("confidence", {})
    verification_reasons = page_payload.get("verification_reasons", [])

    return f"""Please verify Page {page_number} of this examination paper:

CURRENT EXTRACTED DATA:
- Visual Verification Trigger Reasons: {json.dumps(verification_reasons)}
- Current Confidence Breakdown: {json.dumps(confidence, indent=2)}
- Extracted Exam Metadata: {json.dumps(metadata, indent=2)}
- Detected Sections/Parts: {json.dumps(sections)}
- Detected Instructions ({len(instructions)} items):
{json.dumps(instructions, indent=2)}
- Extracted Questions ({len(questions)} items):
{json.dumps([{"number": q.get("number"), "text": q.get("text", "")[:120], "part": q.get("part"), "marks": q.get("marks")} for q in questions], indent=2)}

OCR RAW TEXT:
\"\"\"
{ocr_text[:4000]}
\"\"\"

CRITICAL VERIFICATION TASKS:
1. SOURCE OF TRUTH: Base all decisions strictly on PRINTED/TYPESET text. Completely ignore handwritten notes, pencil/pen markings, ticks, circles, student answers, or scribbles.
2. EXAM METADATA: Inspect the printed header/masthead on this page. If university, subject, session_code/QP code, exam_year, exam_month, max_marks, or duration are missing or inaccurate in Extracted Exam Metadata, supply corrected values in metadata_corrections.
3. STRUCTURE: Identify if any genuine questions or marks were mistakenly trapped in instructions, missing, or mis-numbered.
Return the structured JSON correction output.
"""


async def verify_page_with_gemini(
    page_number: int,
    image_bytes: Optional[bytes],
    page_payload: Dict[str, Any],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> VisionVerificationResult:
    """
    Asynchronously calls Gemini Vision API for a single flagged page.
    Includes bounded retries with exponential backoff.
    """
    global _MOCK_VISION_HANDLER
    start_time = time.time()

    # Check mock hook first (for automated regression tests without network/API calls)
    if _MOCK_VISION_HANDLER is not None:
        logger.info(f"[Vision] Using mock vision handler for Page {page_number}")
        result = _MOCK_VISION_HANDLER(page_number, page_payload)
        elapsed = time.time() - start_time
        logger.info(f"[Vision] Page {page_number} completed in {elapsed:.2f}s (mocked)")
        return result

    # Check API key configuration
    resolved_api_key = (api_key or GEMINI_API_KEY).strip()
    resolved_model = (model or GEMINI_VISION_MODEL).strip() or "gemini-2.0-flash-lite"

    if not resolved_api_key:
        logger.warning(f"[Vision] Page {page_number}: GEMINI_API_KEY is not configured in .env. Skipping live Vision call.")
        return VisionVerificationResult(
            page_number=page_number,
            is_structure_correct=False,
            confidence=0.50,
            needs_manual_review=True,
            review_reason="GEMINI_API_KEY not configured. Manual review recommended.",
            raw_explanation="Vision verification skipped because GEMINI_API_KEY is unset in backend/.env",
        )

    user_text = _build_user_message(page_number, page_payload)

    parts: list[dict[str, Any]] = []
    if image_bytes:
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": img_b64
            }
        })
    parts.append({"text": user_text})

    request_body = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": parts
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.1,
            "maxOutputTokens": 4096,
        }
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{resolved_model}:generateContent?key={resolved_api_key}"

    last_error: Optional[str] = None
    retries = VISION_MAX_RETRIES

    logger.info(f"[Vision] Page {page_number} request started (model={resolved_model})")

    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=float(VISION_TIMEOUT_SECONDS)) as client:
                resp = await client.post(url, json=request_body)

                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        raise ValueError("No candidates returned from Gemini Vision API")

                    content_parts = candidates[0].get("content", {}).get("parts", [])
                    if not content_parts:
                        raise ValueError("Empty response text from Gemini Vision API")

                    raw_text = content_parts[0].get("text", "").strip()

                    # Strip markdown code blocks if present
                    if raw_text.startswith("```json"):
                        raw_text = raw_text[7:]
                    elif raw_text.startswith("```"):
                        raw_text = raw_text[3:]
                    if raw_text.endswith("```"):
                        raw_text = raw_text[:-3]
                    raw_text = raw_text.strip()

                    parsed_json = json.loads(raw_text)
                    result = VisionVerificationResult.model_validate(parsed_json)
                    result.page_number = page_number

                    elapsed = time.time() - start_time
                    logger.info(
                        f"[Vision] Page {page_number} completed in {elapsed:.2f}s "
                        f"(corrections: {len(result.corrections)}, "
                        f"q_corrections: {len(result.question_corrections)}, "
                        f"confidence: {result.confidence:.2f})"
                    )
                    return result

                elif resp.status_code in (429, 503, 500):
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    logger.warning(
                        f"[Vision] Page {page_number}: Transient error ({last_error}), "
                        f"attempt {attempt + 1}/{retries + 1}. Retrying..."
                    )
                    await asyncio.sleep(1.5 * (attempt + 1))
                else:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                    logger.error(f"[Vision] Page {page_number}: Gemini API error: {last_error}")
                    break

        except (httpx.TimeoutException, asyncio.TimeoutError) as te:
            last_error = f"Request timed out after {VISION_TIMEOUT_SECONDS}s"
            logger.warning(f"[Vision] Page {page_number}: {last_error}, attempt {attempt + 1}/{retries + 1}")
            if attempt < retries:
                await asyncio.sleep(1.0 * (attempt + 1))
        except json.JSONDecodeError as je:
            last_error = f"Malformed JSON from Gemini: {je}"
            logger.error(f"[Vision] Page {page_number}: {last_error}")
            break
        except Exception as e:
            last_error = str(e)
            logger.warning(f"[Vision] Page {page_number}: Unexpected error {e}, attempt {attempt + 1}/{retries + 1}")
            if attempt < retries:
                await asyncio.sleep(1.0 * (attempt + 1))

    # All retries exhausted or non-retryable failure
    elapsed = time.time() - start_time
    logger.error(f"[Vision] Page {page_number} failed after {elapsed:.2f}s: {last_error}")
    return VisionVerificationResult(
        page_number=page_number,
        is_structure_correct=False,
        confidence=0.40,
        needs_manual_review=True,
        review_reason=f"Vision verification failed ({last_error})",
        raw_explanation=f"API error: {last_error}",
    )
