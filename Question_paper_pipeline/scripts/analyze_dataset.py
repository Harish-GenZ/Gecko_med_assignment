"""
Dataset Analysis Script — Gecko Med Question Paper Pipeline
============================================================
Inspects all raw datasets (PDFs + images) and generates a detailed
report about:
  - Page count, file sizes
  - Native text vs scanned page ratio
  - Sample text from each document
  - Metadata patterns detected
  - Question numbering patterns
"""

import os
import re
import json
import fitz  # PyMuPDF
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent / "raw_datasets"

# Patterns for common metadata signals
UNIV_PATTERNS = [
    re.compile(r'university', re.I),
    re.compile(r'college', re.I),
    re.compile(r'institute', re.I),
    re.compile(r'BIU|KNRUHS|UHSR', re.I),
]

YEAR_PATTERN   = re.compile(r'\b(20\d{2}|19\d{2})\b')
MONTH_PATTERN  = re.compile(
    r'\b(january|february|march|april|may|june|july|august|september|'
    r'october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b', re.I)
SESSION_PATTERN = re.compile(r'\b(session|code|paper\s*code|roll\s*no|exam\s*code)\b', re.I)

# Question number patterns
Q_NUM_PATTERNS = [
    re.compile(r'^\s*Q\.?\s*\d+', re.M),
    re.compile(r'^\s*\d+\s*[\.\)]\s', re.M),
    re.compile(r'^\s*\(\s*[a-zA-Z]\s*\)\s', re.M),
    re.compile(r'\bQuestion\s+\d+\b', re.I),
]

MCQ_PATTERN = re.compile(r'^\s*\([a-dA-D]\)\s', re.M)

SECTION_PATTERN = re.compile(r'\b(section|part)\s+[A-Z]\b', re.I)

HEADER_KEYWORDS = re.compile(
    r'\b(maximum\s*marks|time\s*allowed|full\s*marks|duration|total\s*marks)\b', re.I)


def analyze_pdf(pdf_path: Path) -> dict:
    """Analyze a single PDF file."""
    result = {
        "file": str(pdf_path.name),
        "size_mb": round(pdf_path.stat().st_size / 1024 / 1024, 2),
        "total_pages": 0,
        "digital_pages": 0,
        "scanned_pages": 0,
        "mixed_pages": 0,
        "avg_chars_per_page": 0,
        "sample_texts": [],
        "years_found": [],
        "months_found": [],
        "section_patterns": [],
        "question_numbering_styles": [],
        "has_mcq": False,
        "has_session_code": False,
        "has_university_mention": False,
        "has_header_keywords": False,
        "potential_paper_boundaries": [],
        "per_page_summary": [],
    }

    try:
        doc = fitz.open(pdf_path)
        result["total_pages"] = len(doc)
        total_chars = 0
        all_text = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text").strip()
            char_count = len(text)
            total_chars += char_count

            # Determine page type
            if char_count > 100:
                page_type = "digital"
                result["digital_pages"] += 1
            elif char_count > 10:
                page_type = "mixed"
                result["mixed_pages"] += 1
            else:
                page_type = "scanned"
                result["scanned_pages"] += 1

            # Check for paper-boundary signals
            boundary_signals = []
            if YEAR_PATTERN.search(text):
                years = YEAR_PATTERN.findall(text)
                result["years_found"].extend(years)
                boundary_signals.append(f"year:{years[0]}")
            if MONTH_PATTERN.search(text):
                m = MONTH_PATTERN.search(text)
                result["months_found"].append(m.group())
                boundary_signals.append(f"month:{m.group()}")
            if SECTION_PATTERN.search(text):
                s = SECTION_PATTERN.search(text)
                result["section_patterns"].append(s.group())
                boundary_signals.append("section_header")
            if SESSION_PATTERN.search(text):
                result["has_session_code"] = True
                boundary_signals.append("session_code")
            if HEADER_KEYWORDS.search(text):
                result["has_header_keywords"] = True
                boundary_signals.append("exam_header")
            for p in UNIV_PATTERNS:
                if p.search(text):
                    result["has_university_mention"] = True
                    boundary_signals.append("university_name")
                    break
            if MCQ_PATTERN.search(text):
                result["has_mcq"] = True
                boundary_signals.append("mcq_options")

            q_styles = []
            for p in Q_NUM_PATTERNS:
                if p.search(text):
                    q_styles.append(p.pattern[:50])
            result["question_numbering_styles"].extend(q_styles)

            if len(boundary_signals) >= 2:
                result["potential_paper_boundaries"].append(page_num)

            page_summary = {
                "page": page_num + 1,
                "type": page_type,
                "chars": char_count,
                "boundary_signals": boundary_signals,
            }
            result["per_page_summary"].append(page_summary)
            all_text.append(text[:300] if text else "")

        result["avg_chars_per_page"] = round(total_chars / max(len(doc), 1))

        # Sample the first 3 pages text
        result["sample_texts"] = [t[:500] for t in all_text[:3] if t]

        # Deduplicate
        result["years_found"] = list(set(result["years_found"]))
        result["months_found"] = list(set(result["months_found"]))
        result["question_numbering_styles"] = list(set(result["question_numbering_styles"]))
        result["section_patterns"] = list(set(result["section_patterns"]))

        doc.close()
    except Exception as e:
        result["error"] = str(e)

    return result


def analyze_image_dir(image_dir: Path) -> dict:
    """Analyze a directory of JPEG images."""
    jpegs = list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.jpg"))
    return {
        "image_directory": str(image_dir),
        "total_images": len(jpegs),
        "total_size_mb": round(sum(f.stat().st_size for f in jpegs) / 1024 / 1024, 2),
        "file_names": [f.name for f in jpegs[:5]] + (["..."] if len(jpegs) > 5 else []),
        "note": "Images need OCR processing. No native text.",
    }


def collect_pdfs(base: Path):
    pdfs = []
    for root, _, files in os.walk(base):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdfs.append(Path(root) / f)
    return pdfs


def collect_image_dirs(base: Path):
    img_dirs = set()
    for root, _, files in os.walk(base):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                img_dirs.add(Path(root))
    return list(img_dirs)


def main():
    report = {
        "base_dir": str(BASE_DIR),
        "pdf_analyses": [],
        "image_dir_analyses": [],
        "summary": {}
    }

    # Analyze PDFs
    pdfs = collect_pdfs(BASE_DIR)
    print(f"Found {len(pdfs)} PDFs")
    for pdf_path in pdfs:
        print(f"  Analyzing: {pdf_path.name} ({pdf_path.stat().st_size / 1024 / 1024:.1f} MB)...")
        analysis = analyze_pdf(pdf_path)
        analysis["full_path"] = str(pdf_path)
        analysis["university_group"] = pdf_path.parts[-3] if len(pdf_path.parts) >= 3 else "unknown"
        report["pdf_analyses"].append(analysis)

    # Analyze image dirs
    img_dirs = collect_image_dirs(BASE_DIR)
    for img_dir in img_dirs:
        if "__MACOSX" in str(img_dir):
            continue
        analysis = analyze_image_dir(img_dir)
        report["image_dir_analyses"].append(analysis)

    # Summary stats
    total_pages = sum(a.get("total_pages", 0) for a in report["pdf_analyses"])
    total_digital = sum(a.get("digital_pages", 0) for a in report["pdf_analyses"])
    total_scanned = sum(a.get("scanned_pages", 0) for a in report["pdf_analyses"])
    total_mixed = sum(a.get("mixed_pages", 0) for a in report["pdf_analyses"])

    report["summary"] = {
        "total_pdfs": len(pdfs),
        "total_image_dirs": len([d for d in img_dirs if "__MACOSX" not in str(d)]),
        "total_pages": total_pages,
        "digital_pages": total_digital,
        "scanned_pages": total_scanned,
        "mixed_pages": total_mixed,
        "pct_scanned": round(100 * total_scanned / max(total_pages, 1), 1),
        "pct_digital": round(100 * total_digital / max(total_pages, 1), 1),
    }

    out_path = Path(__file__).parent / "dataset_analysis_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n=== SUMMARY ===")
    print(json.dumps(report["summary"], indent=2))
    print(f"\nFull report saved to: {out_path}")

    # Print per-PDF highlights
    print("\n=== PER-PDF HIGHLIGHTS ===")
    for a in report["pdf_analyses"]:
        print(f"\n[{a['file']}]")
        print(f"  Pages: {a['total_pages']} | Digital: {a['digital_pages']} | Scanned: {a['scanned_pages']} | Mixed: {a['mixed_pages']}")
        print(f"  Avg chars/page: {a['avg_chars_per_page']}")
        print(f"  Years found: {a['years_found']}")
        print(f"  Months: {a['months_found']}")
        print(f"  Has MCQ: {a['has_mcq']}")
        print(f"  Has sections: {bool(a['section_patterns'])}")
        print(f"  Potential paper boundaries (pages): {a['potential_paper_boundaries'][:20]}")
        print(f"  Q-numbering styles: {a['question_numbering_styles'][:5]}")
        if a.get("sample_texts"):
            print(f"  Sample (pg1): {repr(a['sample_texts'][0][:200])}")


if __name__ == "__main__":
    main()
