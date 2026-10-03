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
Your task is to inspect the uploaded question paper page image against any machine-extracted text, metadata, instructions, and questions.

CRITICAL INSTRUCTIONS:
1. FOREGROUND TARGET DOCUMENT ONLY:
   - If other overlapping papers, background sheets, or desk surfaces are visible behind or around the target question paper (e.g. peeking out at top or sides), extract information ONLY from the main foreground question paper in front.
   - Completely ignore any university names, session codes, dates, or headers from background or overlapping papers (e.g. if the foreground paper is ANQP Code: MBY2P6 / 15th Sept, ignore background paper MBY2P4 / 7th Sept).

2. STRICT SOURCE OF TRUTH — PRINTED / TYPESET DETAILS ONLY:
   - Use ONLY officially printed/typeset text as the source of truth.
   - You MUST COMPLETELY IGNORE all handwritten annotations, pen/pencil markings, roll numbers, or doodles.
   - For MCQs, transcribe only the printed options and text; do NOT choose or modify options based on student circles or ticks.

3. EXAM METADATA INSPECTION & EXTRACTION:
   - Inspect the printed details at the top header/masthead of the foreground question paper:
     * university: Printed university/institution name (e.g. "UHSR", "Kaloji Narayana Rao University", "Bareilly International University").
     * subject: Printed subject/course name (e.g. "Pharmacology (Paper-2)", "Pathology", "Biochemistry").
     * exam_year & exam_month: Printed examination session date/month/year (e.g. "September", "2026").
     * session_code: Printed session code or ANQP code (e.g. "MBY2P6", "MB2019101").
     * paper_code: Printed paper code or QP code (e.g. "1726N", "11016").
     * max_marks: Printed maximum marks / total marks (e.g. "100", "80", "50").
     * duration: Printed time allowed / duration (e.g. "3 hours", "3 Hours").

4. PRINTED INSTRUCTIONS:
   - Extract all officially printed instructions as an array of strings (e.g. items 1, 2, 3, 4).

5. COMPLETE QUESTION EXTRACTION (NO DROPOUTS OR TRUNCATION):
   - Every single question on the page must be extracted completely:
     * For long questions / clinical vignettes (e.g. Q.1), extract the FULL question prompt including the opening scenario AND all sub-parts (a), (b), (c), (d) without cutting off sentences.
     * For short notes (e.g. Q.2), preserve all items (a), (b), (c), (d), (e).
     * For 'Explain why' questions (e.g. Q.3), preserve all items (a), (b), (c).
     * For Multiple Choice Questions (MCQs under Q.4), split them into separate distinct questions ('4.1', '4.2', '4.3'...).
       For each MCQ, provide its prompt in 'text', marks in 'marks' (e.g. '1'), 'type' as 'MCQ', and an 'options' array: [{"label": "a", "text": "..."}, {"label": "b", "text": "..."}].
       DO NOT combine options into the question text.

6. Return strictly valid JSON adhering to the schema below:
{
  "page_number": <int>,
  "is_structure_correct": true,
  "confidence": 0.98,
  "metadata": {
    "university": "<university>",
    "subject": "<subject>",
    "session_code": "<session code e.g. MBY2P6>",
    "paper_code": "<paper code e.g. 1726N>",
    "exam_year": "<year e.g. 2026>",
    "exam_month": "<month e.g. September>",
    "max_marks": "<e.g. 100>",
    "duration": "<e.g. 3 hours>"
  },
  "instructions": [
    "<instruction 1>",
    "<instruction 2>"
  ],
  "questions": [
    {
      "number": "1",
      "heading": "<heading if any>",
      "text": "<full question text with all sub-parts (a), (b)...>",
      "marks": "<marks e.g. 11>",
      "type": "Essay",
      "part": "PART-A",
      "options": null
    },
    {
      "number": "4.1",
      "heading": "Multiple choice questions.",
      "text": "<mcq prompt>",
      "marks": "1",
      "type": "MCQ",
      "part": "PART-A",
      "options": [
        {"label": "a", "text": "<option a text>"},
        {"label": "b", "text": "<option b text>"},
        {"label": "c", "text": "<option c text>"},
        {"label": "d", "text": "<option d text>"}
      ]
    }
  ],
  "needs_manual_review": false,
  "raw_explanation": "<brief summary>"
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

                    # Normalize flexible Gemini outputs:
                    # 1. Convert top-level metadata dict to metadata_corrections if needed
                    if "metadata" in parsed_json and isinstance(parsed_json["metadata"], dict):
                        meta_corr = parsed_json.get("metadata_corrections") or []
                        for k, v in parsed_json["metadata"].items():
                            if v and not any(m.get("field") == k for m in meta_corr):
                                meta_corr.append({"field": k, "value": str(v), "confidence": 0.98})
                        parsed_json["metadata_corrections"] = meta_corr

                    # 2. Convert top-level questions list to question_corrections if needed
                    if "questions" in parsed_json and isinstance(parsed_json["questions"], list):
                        q_corr = parsed_json.get("question_corrections") or []
                        for q in parsed_json["questions"]:
                            q_num = str(q.get("number", "")).replace("Q.", "").strip()
                            if q_num and not any(item.get("question_number") == q_num for item in q_corr):
                                q_corr.append({
                                    "question_number": q_num,
                                    "action": "REPLACE",
                                    "section": q.get("part") or q.get("section") or "PART-A",
                                    "text": q.get("text", ""),
                                    "heading": q.get("heading"),
                                    "marks": str(q.get("marks", "")) if q.get("marks") is not None else None,
                                    "options": q.get("options"),
                                    "type": q.get("type"),
                                    "confidence": 0.98,
                                })
                        parsed_json["question_corrections"] = q_corr

                    if "instructions" in parsed_json and isinstance(parsed_json["instructions"], list):
                        parsed_json["instructions"] = [str(x).strip() for x in parsed_json["instructions"] if str(x).strip()]

                    if "confidence" not in parsed_json or not parsed_json["confidence"]:
                        parsed_json["confidence"] = 0.98

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
