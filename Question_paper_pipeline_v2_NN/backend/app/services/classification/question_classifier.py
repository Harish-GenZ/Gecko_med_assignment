"""
NN Stage 4: Question Type Classifier with Semantic Neural Embeddings.

Classifies each question into one of:
  - Essay
  - Short Notes
  - Very Short Answers
  - MCQ

Hierarchy:
  1. Category-first segmenter signal / MCQ options check (Deterministic, 0.95 conf)
  2. Marks-based thresholds (10+ marks = Essay, 5-7 = Short Notes, 1-4 = VSA, 1 = MCQ)
  3. Section header context (PART-A, Long Questions, etc.)
  4. Explicit regex keyword cues
  5. ★ NN Stage 4: Semantic Neural Embedding Cosine Similarity (MiniLM / BGE-small)
     Classifies questions without explicit keywords in <5ms.
  6. Length fallback (Last resort)
"""

from __future__ import annotations
import re
import logging
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

from app.models.schemas import QuestionType, ReviewStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confidence levels
# ---------------------------------------------------------------------------
HIGH_CONF       = 0.95
SEMANTIC_CONF   = 0.88
MEDIUM_CONF     = 0.80
LOW_CONF        = 0.55
FALLBACK_CONF   = 0.45

# ---------------------------------------------------------------------------
# Keyword patterns
# ---------------------------------------------------------------------------
_ESSAY_RE = re.compile(
    r'\b(describe\s+in\s+detail|enumerate\s+and\s+describe|explain\s+in\s+detail|'
    r'discuss\s+in\s+detail|write\s+an?\s+essay|write\s+in\s+detail|'
    r'classify\s+and\s+describe|what\s+is\s+.*\s+describe|write\s+about\s+.*\s+and\s+describe)\b',
    re.I
)

_SHORT_NOTES_RE = re.compile(
    r'\b(write\s+short\s+note[s]?|brief\s+note[s]?|short\s+note[s]?|'
    r'write\s+about|write\s+briefly|enumerate|list|mention)\b',
    re.I
)

_VERY_SHORT_RE = re.compile(
    r'\b(define|what\s+is|what\s+are|name|give\s+the|state|'
    r'expand|abbreviate|full\s+form|normal\s+value)\b',
    re.I
)

_SECTION_DEFAULTS: Dict[str, QuestionType] = {
    "long question":         QuestionType.ESSAY,
    "long essay":            QuestionType.ESSAY,
    "structured long essay": QuestionType.ESSAY,
    "problem based":         QuestionType.ESSAY,
    "modified essay":        QuestionType.ESSAY,
    "essay":                 QuestionType.ESSAY,
    "short question":        QuestionType.SHORT_NOTES,
    "short notes":           QuestionType.SHORT_NOTES,
    "short answer":          QuestionType.SHORT_NOTES,
    "short essay":           QuestionType.SHORT_NOTES,
    "seq":                   QuestionType.SHORT_NOTES,
    "saq":                   QuestionType.SHORT_NOTES,
    "very short":            QuestionType.VERY_SHORT_ANSWERS,
    "define":                QuestionType.VERY_SHORT_ANSWERS,
    "multiple choice":       QuestionType.MCQ,
    "mcq":                   QuestionType.MCQ,
    "objective":             QuestionType.MCQ,
}


# ---------------------------------------------------------------------------
# NN Stage 4: Semantic Neural Embedding Classifier
# ---------------------------------------------------------------------------
_embedding_model = None
_embedding_init_attempted = False
_ANCHOR_EMBEDDINGS: Dict[QuestionType, np.ndarray] = {}

# Prototypical semantic anchor sentences for cosine matching
_ANCHOR_PROTOTYPES = {
    QuestionType.ESSAY: [
        "Describe in detail the etiology, pathophysiology, clinical features, and management of the patient",
        "Discuss surgical anatomy, operative steps, and complications in detail with a diagram",
        "Classify and describe mechanisms of action, adverse effects, and therapeutic uses",
    ],
    QuestionType.SHORT_NOTES: [
        "Write short notes on pharmacological actions, diagnostic criteria, and clinical features",
        "Briefly explain the differences and differential diagnosis between conditions",
        "Enumerate causes, risk factors, and outline clinical presentations briefly",
    ],
    QuestionType.VERY_SHORT_ANSWERS: [
        "Define the anatomical structure or physiological term precisely",
        "What is the normal reference range or full form",
        "Name two complications, give two examples, or state the law",
    ],
    QuestionType.MCQ: [
        "Which of the following statements is correct regarding the condition",
        "Select the most appropriate option from the choices given below",
    ]
}


def _init_anchor_embeddings(model) -> None:
    """
    Precompute and cache normalized embeddings for anchor prototypes once.
    Eliminates redundant transformer inferences across questions, achieving <2ms latency.
    """
    global _ANCHOR_EMBEDDINGS
    if not _ANCHOR_EMBEDDINGS and model is not None:
        try:
            for q_type, anchors in _ANCHOR_PROTOTYPES.items():
                _ANCHOR_EMBEDDINGS[q_type] = model.encode(anchors, normalize_embeddings=True)
            logger.info("[Classifier] Precomputed semantic anchor embeddings for ultra-fast cosine similarity.")
        except Exception as e:
            logger.warning(f"[Classifier] Failed to precompute anchor embeddings: {e}")


def _get_embedding_model():
    global _embedding_model, _embedding_init_attempted
    if _embedding_init_attempted:
        return _embedding_model
    _embedding_init_attempted = True

    try:
        import importlib
        st_mod = importlib.import_module("sentence_transformers")
        SentenceTransformer = getattr(st_mod, "SentenceTransformer")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("[Classifier] Loaded SentenceTransformer all-MiniLM-L6-v2 for semantic question categorization")
        _init_anchor_embeddings(_embedding_model)
    except Exception as e:
        logger.debug(f"[Classifier] SentenceTransformer not initialized: {e}. Using rule/mark-based categorization.")
    return _embedding_model


def _classify_by_semantic_embedding(text: str) -> Optional[Tuple[QuestionType, float]]:
    """
    Computes semantic similarity against precomputed question type prototypes in <2ms.
    Eliminates reliance on rigid regex keywords without re-encoding anchor sentences.
    """
    model = _get_embedding_model()
    if model is None:
        return None

    global _ANCHOR_EMBEDDINGS
    if not _ANCHOR_EMBEDDINGS:
        _init_anchor_embeddings(model)

    try:
        text_emb = model.encode([text], normalize_embeddings=True)[0]
        best_type = None
        best_score = -1.0

        for q_type, anchor_embs in _ANCHOR_EMBEDDINGS.items():
            similarities = np.dot(anchor_embs, text_emb)
            max_sim = float(np.max(similarities))
            if max_sim > best_score:
                best_score = max_sim
                best_type = q_type

        # Cosine similarity threshold for confident classification
        if best_type and best_score >= 0.50:
            return best_type, SEMANTIC_CONF
    except Exception as emb_err:
        logger.debug(f"[Classifier] Embedding inference error: {emb_err}")

    return None


def _classify_by_marks(marks_str: Optional[str]) -> Optional[Tuple[QuestionType, float]]:
    if not marks_str:
        return None

    num_m = re.search(r'(\d+)', marks_str)
    if not num_m:
        return None

    marks = int(num_m.group(1))
    multi = re.search(r'(\d+)\s*[x×]\s*(\d+)', marks_str, re.I)
    if multi:
        marks = int(multi.group(1))

    if marks == 1:
        return QuestionType.MCQ, HIGH_CONF
    elif marks <= 4:
        return QuestionType.VERY_SHORT_ANSWERS, HIGH_CONF
    elif marks <= 7:
        return QuestionType.SHORT_NOTES, HIGH_CONF
    elif marks >= 10:
        return QuestionType.ESSAY, HIGH_CONF

    return None


def _classify_by_section(section_context: Optional[str]) -> Optional[Tuple[QuestionType, float]]:
    if not section_context:
        return None
    ctx = section_context.lower()
    for key, qtype in _SECTION_DEFAULTS.items():
        if key in ctx:
            return qtype, MEDIUM_CONF
    return None


def classify_question(
    question_data: Dict[str, Any],
    section_context: Optional[str] = None,
) -> Tuple[QuestionType, float]:
    """
    Master question classifier.
    Combines rule-based certainty with NN Stage 4 semantic embeddings.
    """
    # Rule 0: Pre-detected category from segmenter
    cat_str = question_data.get("category") or question_data.get("type")
    if cat_str:
        for q_type in QuestionType:
            if q_type.value.lower() == str(cat_str).lower():
                return q_type, HIGH_CONF

    text = question_data.get("text", "")
    options = question_data.get("options", [])
    marks = question_data.get("marks")

    # Rule 1: MCQ options present
    if options and len(options) >= 2:
        return QuestionType.MCQ, HIGH_CONF

    # Rule 2: Marks-based classification
    marks_result = _classify_by_marks(marks)
    if marks_result:
        return marks_result

    # Rule 3: Section context
    section_result = _classify_by_section(section_context or question_data.get("section"))

    # Rule 4: Regex keyword matching
    keyword_result = _classify_by_keywords(text)

    if section_result and keyword_result:
        sk_type, sk_conf = keyword_result
        if sk_conf >= section_result[1]:
            return keyword_result
        return section_result
    elif section_result:
        return section_result
    elif keyword_result:
        return keyword_result

    # ★ Rule 5 (NN Stage 4): Semantic Neural Embeddings
    semantic_result = _classify_by_semantic_embedding(text)
    if semantic_result:
        return semantic_result

    # Rule 6: Heuristic fallback by question text length
    return _classify_by_length(text)


def _classify_by_keywords(text: str) -> Optional[Tuple[QuestionType, float]]:
    if _ESSAY_RE.search(text):
        return QuestionType.ESSAY, MEDIUM_CONF
    if _SHORT_NOTES_RE.search(text):
        return QuestionType.SHORT_NOTES, MEDIUM_CONF
    if _VERY_SHORT_RE.search(text):
        return QuestionType.VERY_SHORT_ANSWERS, MEDIUM_CONF
    return None


def _classify_by_length(text: str) -> Tuple[QuestionType, float]:
    word_count = len(text.split())
    if word_count <= 10:
        return QuestionType.VERY_SHORT_ANSWERS, FALLBACK_CONF
    elif word_count <= 30:
        return QuestionType.SHORT_NOTES, FALLBACK_CONF
    else:
        return QuestionType.ESSAY, FALLBACK_CONF
