import re
from typing import NamedTuple

# Positive retail outlet, supermarket, and general store keywords (English)
RETAIL_KEYWORDS_EN = {
    "supermarket", "super market", "general store", "general stores", "store", "stores",
    "mart", "hypermarket", "grocery", "groceries", "provision", "provisions",
    "departmental", "department store", "bazaar", "traders", "kirana", "retail",
    "daily needs", "organics", "fresh", "super mart", "super center", "wholesale",
    "fancy store", "stationery", "enterprise", "enterprises", "agency", "agencies",
    # Also valid retail subsectors
    "pharmacy", "medicals", "medical", "chemist", "bakery", "sweets"
}

# Positive retail outlet, supermarket, and general store keywords (Tamil)
RETAIL_KEYWORDS_TA = {
    "சூப்பர் மார்க்கெட்", "ஜெனரல் ஸ்டோர்ஸ்", "மளிகை", "மளிகை கடை", "டிபார்ட்மென்டல்",
    "டிபார்ட்மெண்ட்", "ஸ்டோர்ஸ்", "ஸ்டோர்", "பல்பொருள் அங்காடி", "அங்காடி", "டிரேடர்ஸ்",
    "கடை", "வியாபாரம்", "ஏஜென்சி", "மருந்தகம்", "மெடிக்கல்ஸ்"
}

# Negative non-store / invalid input patterns (English)
NON_OUTLET_KEYWORDS_EN = {
    "selfie", "my photo", "portrait", "bedroom", "living room", "kitchen", "bathroom",
    "empty road", "highway", "forest", "beach", "scenery", "landscape", "test dummy",
    "random", "sample test", "asdf"
}

# Negative non-store / invalid input patterns (Tamil)
NON_OUTLET_KEYWORDS_TA = {
    "வீடு", "படுக்கையறை", "காடு", "ரோடு", "சாலை"
}


class NameClassificationResult(NamedTuple):
    is_valid_outlet_name: bool
    confidence: float
    category: str
    detected_keywords: list[str]
    rejection_reason: str | None


class NameClassifier:
    """
    Evaluates whether an outlet name semantically corresponds to a retail outlet
    (supermarket, general store, grocery/provision shop, mart, retail store)
    or represents an invalid non-store input.
    Supports English and Tamil names.
    """

    @classmethod
    def classify(cls, name: str) -> NameClassificationResult:
        if not name or not name.strip():
            return NameClassificationResult(
                is_valid_outlet_name=False,
                confidence=1.0,
                category="EMPTY",
                detected_keywords=[],
                rejection_reason="Outlet name is empty.",
            )

        text_lower = name.strip().lower()

        # Tokenize by whitespace and non-alphanumeric (for Latin/Tamil scripts)
        words = re.findall(r"[\w\u0B80-\u0BFF]+", text_lower)
        words_set = set(words)

        # 1. Check for explicit NON-outlet / invalid terms
        detected_invalid: list[str] = []
        for kw in NON_OUTLET_KEYWORDS_EN:
            if " " in kw:
                if kw in text_lower:
                    detected_invalid.append(kw)
            elif kw in words_set:
                detected_invalid.append(kw)

        for kw in NON_OUTLET_KEYWORDS_TA:
            if kw in text_lower:
                detected_invalid.append(kw)

        if detected_invalid:
            return NameClassificationResult(
                is_valid_outlet_name=False,
                confidence=0.95,
                category="NON_OUTLET",
                detected_keywords=detected_invalid,
                rejection_reason=f"Name contains non-outlet term(s): {', '.join(detected_invalid)}",
            )

        # 2. Check for positive retail store / supermarket keywords
        detected_retail: list[str] = []
        for kw in RETAIL_KEYWORDS_EN:
            if " " in kw:
                if kw in text_lower:
                    detected_retail.append(kw)
            elif kw in words_set:
                detected_retail.append(kw)

        for kw in RETAIL_KEYWORDS_TA:
            if kw in text_lower:
                detected_retail.append(kw)

        if detected_retail:
            return NameClassificationResult(
                is_valid_outlet_name=True,
                confidence=0.95,
                category="RETAIL_STORE",
                detected_keywords=detected_retail,
                rejection_reason=None,
            )

        # Standard business / outlet naming pattern (e.g. "Sri Murugan Trading" or "Annamalai Traders" or "Nilgiris")
        return NameClassificationResult(
            is_valid_outlet_name=True,
            confidence=0.75,
            category="GENERAL_OUTLET",
            detected_keywords=[],
            rejection_reason=None,
        )
