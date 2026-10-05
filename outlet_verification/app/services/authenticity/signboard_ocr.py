import difflib
import logging
import re
import threading
from typing import Any, NamedTuple
import numpy as np

from PIL import Image
from app.services.authenticity.utils import load_pil_image

logger = logging.getLogger("outlet_verification.authenticity.ocr")

RETAIL_SIGNBOARD_KEYWORDS = {
    "supermarket", "super market", "general store", "general stores", "store", "stores",
    "mart", "provisions", "provision", "grocery", "groceries", "departmental",
    "department store", "hypermarket", "retail", "bazaar", "traders", "kirana",
    "daily needs", "foods", "organics", "wholesale", "agency", "enterprises",
    "pharmacy", "medicals", "medical", "chemist", "druggist", "bakery", "sweets",
    "மருந்தகம்", "மெடிக்கல்ஸ்", "பார்மசி", "மளிகை", "சூப்பர் மார்க்கெட்", "ஸ்டோர்ஸ்", "கடை"
}

# Domain Category Synonym Clusters
PHARMACY_CLUSTER = {
    "pharmacy", "medical", "medicals", "chemist", "druggist", "drugstore",
    "marunthagam", "baarmasi", "medikkals", "medikals", "மருந்தகம்", "மெடிக்கல்ஸ்", "பார்மசி"
}

GROCERY_CLUSTER = {
    "supermarket", "super market", "hypermarket", "grocery", "groceries",
    "provision", "provisions", "kirana", "மளிகை", "சூப்பர் மார்க்கெட்"
}

GENERAL_STORE_CLUSTER = {
    "general store", "general stores", "store", "stores", "mart", "bazaar",
    "shop", "traders", "agency", "enterprises", "ஸ்டோர்", "ஸ்டோர்ஸ்", "கடை", "அங்காடி"
}

SWEET_BAKERY_CLUSTER = {
    "sweets", "sweet", "bakery", "bakes", "cake", "cakes", "snack", "snacks", "ஸ்வீட்ஸ்", "பேக்கரி"
}

INCIDENTAL_SIGNAGE_TERMS = {
    # Food & Beverages sold / advertised
    "healthy", "drinks", "drink", "cool", "tea", "coffee", "ice", "cream", "icecream",
    "snacks", "juice", "fresh", "water", "beverages", "milk", "dairy", "bakery",
    "biscuits", "chocolates", "cadbury", "amul", "horlicks", "boost", "complan",
    # Operational & Courtesy
    "welcome", "open", "closed", "entry", "exit", "push", "pull", "thank", "you",
    "visit", "again", "please", "parking", "no parking", "cctv", "camera", "surveillance",
    "24x7", "24 hours", "timing", "timings", "working hours",
    # Payments & Digital
    "gpay", "google pay", "phonepe", "paytm", "bhim", "upi", "qr", "scan", "accepted",
    "cards", "visa", "mastercard", "debit", "credit", "atm", "xerox", "photocopy",
    "recharge", "mobile",
    # Health & Safety
    "mask", "sanitize", "sanitizer", "social", "distancing",
    # Regulatory & Contact prefixes
    "tel", "phone", "cell", "mob", "call", "gst", "gstin", "reg", "dl", "licence", "license"
}

COMMON_STOPWORDS_AND_FRAGMENTS = {
    "the", "and", "for", "with", "all", "our", "you", "your", "thy", "new",
    "sri", "shri", "om", "near", "opp", "opposite", "road", "street", "st",
    "rd", "main", "cross", "bldg", "building", "floor", "ground", "shop", "no",
    "nag", "nagar", "colony", "city", "town", "dist", "district", "pin", "code"
}

# Established commercial retail and pharmacy chains for genuine conflicting brand detection
KNOWN_COMMERCIAL_CHAINS = {
    # Established Pharmacy Chains
    "apollo", "medplus", "wellness forever", "guardian", "netmeds", "pharmeasy", "frank ross", "sanjivani",
    # Food & Retail Chains
    "subway", "mcdonalds", "kfc", "dominos", "pizza hut", "burger king",
    "starbucks", "cafe coffee day", "costa coffee", "dunkin",
    # Supermarkets & Department Stores
    "dmart", "reliance", "spencer", "spencers", "more supermarket", "nature basket",
    "big bazaar", "bigbazaar", "easyday", "7-eleven", "spar",
    # Electronics & Eyewear
    "croma", "vijay sales", "titan", "lenskart", "bata"
}


# Unicode mappings for Tamil script phonetic transliteration
TAMIL_VOWELS = {
    0x0B85: "a", 0x0B86: "aa", 0x0B87: "i", 0x0B88: "ee", 0x0B89: "u",
    0x0B8A: "oo", 0x0B8E: "e", 0x0B8F: "ae", 0x0B90: "ai", 0x0B92: "o",
    0x0B93: "oa", 0x0B94: "au",
}

TAMIL_CONSONANTS = {
    0x0B95: "k", 0x0B99: "ng", 0x0B9A: "s", 0x0B9C: "j", 0x0B9E: "nj",
    0x0B9F: "d", 0x0BA3: "n", 0x0BA4: "th", 0x0BA8: "n", 0x0BA9: "n",
    0x0BAA: "b", 0x0BAE: "m", 0x0BAF: "y", 0x0BB0: "r", 0x0BB1: "r",
    0x0BB2: "l", 0x0BB3: "l", 0x0BB4: "zh", 0x0BB5: "v", 0x0BB6: "sh",
    0x0BB7: "sh", 0x0BB8: "s", 0x0BB9: "h",
}

TAMIL_VOWEL_SIGNS = {
    0x0BBE: "a", 0x0BBF: "i", 0x0BC0: "ee", 0x0BC1: "u", 0x0BC2: "oo",
    0x0BC6: "e", 0x0BC7: "e", 0x0BC8: "ai", 0x0BCA: "o", 0x0BCB: "o",
    0x0BCC: "au",
}

VIRAMA = 0x0BCD  # Tamil pulli (suppresses default vowel 'a')


def transliterate_tamil(text: str) -> str:
    """
    Converts Tamil Unicode text to clean Latin phonetic representation.
    e.g. 'பர்மா மெடிக்கல்ஸ்' -> 'barma medikkals'
    """
    if not text:
        return ""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        cp = ord(text[i])
        if cp in TAMIL_VOWELS:
            out.append(TAMIL_VOWELS[cp])
            i += 1
        elif cp in TAMIL_CONSONANTS:
            base = TAMIL_CONSONANTS[cp]
            if i + 1 < n and ord(text[i + 1]) == VIRAMA:
                out.append(base)
                i += 2
            elif i + 1 < n and ord(text[i + 1]) in TAMIL_VOWEL_SIGNS:
                out.append(base + TAMIL_VOWEL_SIGNS[ord(text[i + 1])])
                i += 2
            else:
                out.append(base + "a")
                i += 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


class OCRAnalysisResult(NamedTuple):
    extracted_text_lines: list[str]
    detected_retail_keywords: list[str]
    name_match_score: float | None
    has_readable_signboard: bool
    is_cross_lingual_or_category_match: bool = False
    is_incidental_signage: bool = False
    has_conflicting_brand: bool = False
    conflicting_brand_name: str | None = None
    is_unclear_or_multilingual: bool = False
    clarity_status: str = "LEGIBLE"  # "LEGIBLE", "UNCLEAR_OR_BLURRY", "MULTILINGUAL_OR_STYLIZED", "NO_SIGNBOARD"
    rejection_reason: str | None = None


class SignboardOCRService:
    """
    Signboard OCR and semantic cross-verification service for retail outlets and pharmacies.
    Extracts text from the storefront photograph using EasyOCR, supports Tamil-English
    cross-lingual matching, domain synonym expansion, brand extraction, incidental signage
    filtering, and semantic embeddings.
    """
    _instance: "SignboardOCRService | None" = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self.reader: Any = None
        self._is_loaded: bool = False

    @classmethod
    def get_instance(cls) -> "SignboardOCRService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load_reader(self) -> None:
        if self._is_loaded:
            return

        with self._lock:
            if self._is_loaded:
                return

            try:
                import warnings
                warnings.filterwarnings("ignore", category=UserWarning)
                import easyocr
                logger.info("Initializing EasyOCR reader for signboard text extraction...")
                self.reader = easyocr.Reader(["en"], gpu=False, verbose=False, quantize=False)
                self._is_loaded = True
                logger.info("EasyOCR reader successfully initialized.")
            except Exception as exc:
                logger.warning("Failed to initialize EasyOCR reader: %s. OCR checks will be bypassed.", exc)
                self.reader = None
                self._is_loaded = True

    def analyze_image(self, image_input: Any, submitted_name: str | None = None) -> OCRAnalysisResult:
        """
        Extracts text from the image and performs multimodal, cross-lingual,
        and semantic name cross-matching with incidental signage filtering.
        Optimized for swift CPU inference via automatic dimension downscaling and
        noise / blur / multilingual detection.
        """
        self.load_reader()
        if self.reader is None:
            return OCRAnalysisResult(
                extracted_text_lines=[],
                detected_retail_keywords=[],
                name_match_score=None,
                has_readable_signboard=False,
                is_cross_lingual_or_category_match=False,
                is_incidental_signage=False,
                has_conflicting_brand=False,
                conflicting_brand_name=None,
                is_unclear_or_multilingual=True,
                clarity_status="UNCLEAR_OR_BLURRY",
                rejection_reason=None,
            )

        pil_img = load_pil_image(image_input)

        # 1. Blur & Sharpness estimation (Laplacian variance approximation)
        is_blurry = False
        try:
            gray = np.array(pil_img.convert("L"), dtype=np.float32)
            if gray.shape[0] > 10 and gray.shape[1] > 10:
                laplacian = (
                    gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
                    - 4.0 * gray[1:-1, 1:-1]
                )
                is_blurry = float(laplacian.var()) < 40.0
        except Exception:
            pass

        # 2. Downscale high-resolution images to max 1200px to prevent 40+ second CPU stalls
        MAX_OCR_DIM = 1200
        w, h = pil_img.size
        if max(w, h) > MAX_OCR_DIM:
            scale = MAX_OCR_DIM / float(max(w, h))
            pil_ocr = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
        else:
            pil_ocr = pil_img

        img_np = np.array(pil_ocr)

        try:
            results = self.reader.readtext(img_np)
        except Exception as exc:
            logger.warning("EasyOCR execution failed: %s", exc)
            return OCRAnalysisResult(
                extracted_text_lines=[],
                detected_retail_keywords=[],
                name_match_score=None,
                has_readable_signboard=False,
                is_cross_lingual_or_category_match=False,
                is_incidental_signage=False,
                has_conflicting_brand=False,
                conflicting_brand_name=None,
                is_unclear_or_multilingual=True,
                clarity_status="UNCLEAR_OR_BLURRY",
                rejection_reason=None,
            )

        extracted_lines: list[str] = []
        high_conf_words: dict[str, float] = {}
        detected_confidences: list[float] = []
        has_regional_script = False

        for _, text, conf in results:
            clean_t = text.strip() if text else ""
            if not clean_t:
                continue
            c_val = float(conf)
            detected_confidences.append(c_val)

            # Detect regional Tamil / Devanagari Unicode scripts
            if re.search(r"[\u0B80-\u0BFF\u0900-\u097F]", clean_t):
                has_regional_script = True

            # Strip edge punctuation and noise like underscores or hyphens: "JG_l_" -> "JG_l"
            stripped_t = re.sub(r"^[\W_]+|[\W_]+$", "", clean_t)
            if not stripped_t:
                continue

            # Require at least 3 alphanumeric characters
            alpha_tokens = re.findall(r"[a-zA-Z0-9\u0B80-\u0BFF]", stripped_t)
            if len(alpha_tokens) < 3:
                continue

            # Must have confidence >= 0.45 for short words (< 5 chars) and >= 0.35 for longer words
            min_conf = 0.35 if len(stripped_t) >= 5 else 0.45
            if c_val < min_conf:
                continue

            # Check that line contains real pronounceable words (not consonant smashes like 'jgl')
            words_in_line = stripped_t.lower().split()
            filtered_line_words = []
            for w in words_in_line:
                clean_w = re.sub(r"[^a-z0-9\u0B80-\u0BFF]", "", w)
                if not clean_w:
                    continue
                # If Latin, check for vowel or digits (allow common 2-letter abbreviations)
                if re.match(r"^[a-z]+$", clean_w):
                    if len(clean_w) == 2 and clean_w not in {"st", "dr", "mr", "tv", "rd", "no", "co"}:
                        if not re.search(r"[aeiouy]", clean_w):
                            continue
                    elif len(clean_w) >= 3:
                        if not re.search(r"[aeiouy]", clean_w):
                            continue
                filtered_line_words.append(clean_w)
                if len(clean_w) >= 3 and not clean_w.isdigit():
                    high_conf_words[clean_w] = max(high_conf_words.get(clean_w, 0.0), c_val)

            if filtered_line_words:
                extracted_lines.append(stripped_t)

        total_detections = len(results)
        valid_lines_count = len(extracted_lines)
        avg_conf = (sum(detected_confidences) / len(detected_confidences)) if detected_confidences else 0.0

        is_unclear_or_multilingual = False
        clarity_status = "LEGIBLE"

        if has_regional_script:
            is_unclear_or_multilingual = True
            clarity_status = "MULTILINGUAL_OR_STYLIZED"
        elif total_detections > 0 and valid_lines_count == 0:
            # Text was detected by CRAFT but could not be extracted with high confidence
            is_unclear_or_multilingual = True
            clarity_status = "LOW_CONFIDENCE_INCONCLUSIVE" if not is_blurry else "UNCLEAR_OR_BLURRY"
        elif is_blurry and valid_lines_count < 2:
            is_unclear_or_multilingual = True
            clarity_status = "UNCLEAR_OR_BLURRY"
        elif valid_lines_count == 0:
            clarity_status = "NO_SIGNBOARD"

        full_ocr_raw = " ".join(extracted_lines).lower()
        full_ocr_trans = transliterate_tamil(full_ocr_raw).lower()
        all_ocr_text = f"{full_ocr_raw} {full_ocr_trans}"

        raw_words = set(re.findall(r"[\w\u0B80-\u0BFF]+", full_ocr_raw))
        trans_words = set(re.findall(r"[a-z0-9]+", full_ocr_trans))
        all_ocr_words = raw_words.union(trans_words)

        # 1. Detect retail keywords (supports English and Tamil terms)
        detected_retail = [
            kw for kw in RETAIL_SIGNBOARD_KEYWORDS
            if kw in all_ocr_words or kw in all_ocr_text
        ]

        # 2. Semantic Cross-Match with submitted name
        name_match_score: float | None = None
        is_semantic_match = False
        is_incidental_signage = False
        has_conflicting_brand = False
        conflicting_brand_name: str | None = None

        if submitted_name and extracted_lines:
            sub_raw = submitted_name.strip().lower()
            sub_trans = transliterate_tamil(sub_raw).lower()
            sub_words = set(re.findall(r"[a-z0-9]+", sub_trans))

            # A. Sequence matching
            seq_score = max(
                difflib.SequenceMatcher(None, sub_raw, full_ocr_raw).ratio(),
                difflib.SequenceMatcher(None, sub_trans, full_ocr_trans).ratio(),
            )

            # B. Token overlap
            overlap_words = sub_words.intersection(all_ocr_words)
            token_overlap_score = len(overlap_words) / max(1, len(sub_words))

            # C. Category Synonym Matching (e.g. "pharmacy" <=> "medicals" <=> "மருந்தகம்")
            all_clusters = [PHARMACY_CLUSTER, GROCERY_CLUSTER, GENERAL_STORE_CLUSTER, SWEET_BAKERY_CLUSTER]
            category_matched = False
            for cluster in all_clusters:
                if sub_words.intersection(cluster) and all_ocr_words.intersection(cluster):
                    category_matched = True
                    break

            # D. Incidental Signage Detection
            # Filter words to determine if OCR captured only product promotions or non-brand slogans
            ocr_meaningful_words = {w for w in trans_words if len(w) > 1 and not w.isdigit()}
            ocr_non_incidental = {w for w in ocr_meaningful_words if w not in INCIDENTAL_SIGNAGE_TERMS}

            if ocr_meaningful_words and len(ocr_non_incidental) == 0:
                is_incidental_signage = True

            # E. Brand Name Phonetic & Decomposition Matching
            all_cluster_words = set().union(*all_clusters)
            sub_brands = sub_words - all_cluster_words - INCIDENTAL_SIGNAGE_TERMS
            ocr_brands = ocr_non_incidental - all_cluster_words

            brand_score = 0.0
            if sub_brands:
                for sb in sub_brands:
                    for ob in ocr_brands:
                        # Direct string ratio
                        r = difflib.SequenceMatcher(None, sb, ob).ratio()
                        # Vowel-collapsed phonetic comparison (e.g., 'burma' -> 'brm', 'barma' -> 'brm')
                        sb_c = re.sub(r"[aeiou]+", "", sb)
                        ob_c = re.sub(r"[aeiou]+", "", ob)
                        if sb_c == ob_c and len(sb_c) >= 2:
                            r = max(r, 0.90)
                        elif len(sb_c) >= 2 and len(ob_c) >= 2 and (sb_c in ob_c or ob_c in sb_c):
                            r = max(r, 0.85)
                        elif sb in ob or ob in sb:
                            r = max(r, 0.85)
                        brand_score = max(brand_score, r)
            else:
                brand_score = 1.0 if category_matched else 0.0

            # F. Dense Semantic Embedding Similarity (all-MiniLM-L6-v2)
            embedding_sim = 0.0
            try:
                from app.services.embeddings import generate_name_embedding
                e_sub = np.array(generate_name_embedding(sub_raw))
                e_ocr = np.array(generate_name_embedding(full_ocr_raw))
                embedding_sim = max(0.0, float(np.dot(e_sub, e_ocr)))
                # Also test transliterated text
                if full_ocr_trans != full_ocr_raw:
                    e_trans = np.array(generate_name_embedding(full_ocr_trans))
                    embedding_sim = max(embedding_sim, float(np.dot(e_sub, e_trans)))
            except Exception as e_exc:
                logger.debug("Semantic embedding comparison bypassed: %s", e_exc)

            # G. Conflicting Commercial Brand Detection
            # A conflicting commercial brand requires high confidence (>= 0.70), matching an established
            # commercial chain brand (e.g., 'medplus' when submitted 'apollo'), and having no phonetic/semantic match.
            conflicting_tokens: list[str] = []
            if ocr_brands and not is_incidental_signage and not is_unclear_or_multilingual:
                for ob in ocr_brands:
                    if (
                        len(ob) >= 4
                        and ob in KNOWN_COMMERCIAL_CHAINS
                        and high_conf_words.get(ob, 0.0) >= 0.70
                    ):
                        matched = any(
                            difflib.SequenceMatcher(None, sb, ob).ratio() >= 0.65
                            or (re.sub(r"[aeiou]+", "", sb) == re.sub(r"[aeiou]+", "", ob) and len(re.sub(r"[aeiou]+", "", sb)) >= 2)
                            or sb in ob or ob in sb
                            for sb in sub_brands
                        )
                        if not matched:
                            conflicting_tokens.append(ob)

                if conflicting_tokens and brand_score < 0.50 and valid_lines_count >= 1:
                    has_conflicting_brand = True
                    conflicting_brand_name = " ".join(conflicting_tokens)

            # H. Evidence Synthesis
            if category_matched and brand_score >= 0.70:
                name_match_score = max(0.85, brand_score)
                is_semantic_match = True
            elif brand_score >= 0.80:
                name_match_score = brand_score
                is_semantic_match = True
            elif embedding_sim >= 0.65:
                name_match_score = embedding_sim
                is_semantic_match = True
            elif is_incidental_signage and not has_conflicting_brand:
                # Incidental promotional signage without competing brand
                name_match_score = 0.85
                is_semantic_match = True
            elif max(token_overlap_score, seq_score, embedding_sim, brand_score) >= 0.40 and not has_conflicting_brand:
                name_match_score = round(max(token_overlap_score, seq_score, embedding_sim, brand_score), 4)
                is_semantic_match = True
            else:
                # OCR could not extract clear name details with sufficient confidence
                name_match_score = None
                is_semantic_match = False
                if not has_conflicting_brand:
                    is_unclear_or_multilingual = True
                    clarity_status = "LOW_CONFIDENCE_INCONCLUSIVE"

        has_readable_signboard = len(extracted_lines) > 0 and not is_unclear_or_multilingual
        if is_unclear_or_multilingual or not has_readable_signboard:
            if not has_conflicting_brand:
                name_match_score = None

        return OCRAnalysisResult(
            extracted_text_lines=extracted_lines,
            detected_retail_keywords=detected_retail,
            name_match_score=name_match_score,
            has_readable_signboard=has_readable_signboard,
            is_cross_lingual_or_category_match=is_semantic_match,
            is_incidental_signage=is_incidental_signage,
            has_conflicting_brand=has_conflicting_brand,
            conflicting_brand_name=conflicting_brand_name,
            is_unclear_or_multilingual=is_unclear_or_multilingual,
            clarity_status=clarity_status,
            rejection_reason=None,
        )


