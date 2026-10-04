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
import re
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


def _norm_sec(s: Optional[str]) -> str:
    """Normalize section label for robust matching, e.g. 'Sectiom A' -> 'SECTIONA'."""
    if not s:
        return ""
    clean = re.sub(r'[\s\(\)\[\]\-_:]', '', str(s)).upper()
    clean = clean.replace("SECTIOM", "SECTION").replace("SECTON", "SECTION")
    clean = clean.replace("PART", "SECTION").replace("GROUP", "SECTION")
    return clean


SYSTEM_PROMPT = """You are an expert medical examination paper proofreader and quality verification specialist.
Your primary role is to inspect the uploaded question paper page image against the machine-extracted OCR text and questions, and CORRECT ALL SPELLING MISTAKES, TYPOS, MERGED WORDS, AND OCR DEFECTS while strictly preserving all questions and structure.

CORE RESPONSIBILITIES:
1. SPELLING & TYPO CORRECTION (HIGHEST PRIORITY):
   - Correct all misspelled English and medical terms, e.g.:
     * "scemario" -> "scenario", "folawing" -> "following", "iksed" -> "used", "Lomg" -> "Long", "Questioms" -> "Questions", "Sectiom" -> "Section", "Modifned" -> "Modified", "Deseribe" -> "Describe", "Eirumerate" -> "Enumerate", "epideimig" -> "epidemic", "epidenniological" -> "epidemiological", "healh" -> "health", "Bhorc Commitce" -> "Bhore Committee", "developnent" -> "development".
   - Fix merged words with missing spaces caused by OCR scanners, e.g.:
     * "AsthemedicalofficerofaPHC" -> "As the medical officer of a PHC"
     * "PrepareaprojecttoimprovetheservicesoftheAnganwariworker" -> "Prepare a project to improve the services of the Anganwadi worker"
     * "bDifferentiateBetween objectiveandgoalinhealhplanning" -> "b. Differentiate between objective and goal in health planning"
     * "DifferentiateBetweenCostbeneftandcosteffectiveanalysis" -> "c. Differentiate between cost benefit and cost effective analysis"
     * "DifferentiateBetweenProgrammeevaluationandreviewtechniqueandcriticalpathmethod" -> "d. Differentiate between programme evaluation and review technique and critical path method"
   - Fix broken sub-part labels and OCR noise artifacts (e.g. restoring faint text from the printed image).

2. DO NOT DROP OR REMOVE QUESTIONS:
   - You MUST output every single question provided in Extracted Questions, plus any additional questions clearly visible in the image.
   - Maintain all sections (e.g. Section A, Section B). DO NOT merge Section B into Section A or drop Section B.
   - For every question, keep its sub-parts (a, b, c, d, e...) clearly formatted in 'text'.

3. EXAM METADATA & INSTRUCTIONS:
   - Extract and correct header metadata (university, subject, exam_year, exam_month, session_code, paper_code, max_marks, duration).
   - If university name is printed in header (e.g. "Rohilkhand Medical College & Hospital, Bareilly" or "Bareilly International University"), ensure it is accurately captured.
   - Fix any typos in printed instructions.
   - Base all decisions strictly on PRINTED/TYPESET text. Ignore handwritten annotations, roll numbers, or doodles.

4. OUTPUT FORMAT:
Return strictly valid JSON adhering to this schema:
{
  "page_number": <int>,
  "is_structure_correct": true,
  "confidence": 0.98,
  "metadata": {
    "university": "<university>",
    "subject": "<subject>",
    "session_code": "<session code or ANQP code>",
    "paper_code": "<paper code>",
    "exam_year": "<year>",
    "exam_month": "<month>",
    "max_marks": "<marks>",
    "duration": "<duration>"
  },
  "instructions": [
    "<corrected instruction 1>",
    "<corrected instruction 2>"
  ],
  "questions": [
    {
      "number": "<question number e.g. 1, 2>",
      "part": "<section name e.g. Section A, Section B>",
      "heading": "<heading if any e.g. Structured Long Essay Questions>",
      "text": "<full question text with all sub-parts (a), (b)... spelling and spacing fully corrected>",
      "marks": "<marks e.g. 10, 30>",
      "type": "<Essay | Short Notes | Very Short Answers | MCQ>",
      "options": null
    }
  ],
  "needs_manual_review": false,
  "raw_explanation": "<brief summary of spelling/typos corrected>"
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

    formatted_questions = []
    for q in questions:
        formatted_questions.append({
            "number": str(q.get("number", "")),
            "part": q.get("part") or q.get("section") or "Section A",
            "heading": q.get("heading"),
            "marks": q.get("marks"),
            "text": q.get("text", ""),
        })

    return f"""Please review and proofread Page {page_number} of this medical examination paper:

CURRENT EXTRACTED DATA:
- Header Metadata: {json.dumps(metadata, indent=2)}
- Detected Sections: {json.dumps(sections)}
- Detected Instructions ({len(instructions)} items):
{json.dumps(instructions, indent=2)}
- Extracted Questions ({len(questions)} items to proofread and correct):
{json.dumps(formatted_questions, indent=2)}

OCR RAW TEXT:
\"\"\"
{ocr_text[:4000]}
\"\"\"

PROOFREADING & SPELL-CHECKING INSTRUCTIONS:
1. Proofread and correct all spelling mistakes, typos, character glitches, and merged words in the question texts, headings, sections, and header metadata.
2. Every question provided above in "Extracted Questions" MUST be included in the output JSON with its spelling and wording corrected. Preserve all questions across all sections (e.g. Section A, Section B). Do NOT drop or delete any question.
3. Return the fully proofread output as valid JSON matching the schema.
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
            review_reason="Vision verification API key not configured. Manual review recommended.",
            raw_explanation="Vision verification skipped because vision API key is unset in configuration.",
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
                            q_sec = q.get("part") or q.get("section") or "Section A"
                            q_sec_norm = _norm_sec(q_sec)
                            if q_num and not any(
                                item.get("question_number") == q_num and _norm_sec(item.get("section")) == q_sec_norm
                                for item in q_corr
                            ):
                                q_corr.append({
                                    "question_number": q_num,
                                    "action": "REPLACE",
                                    "section": q_sec,
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
