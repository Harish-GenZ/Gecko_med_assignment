"""
OCR engine wrapper using PaddleOCR.

Returns structured OCR results with text, bounding boxes, and confidence.

Design decisions:
- PaddleOCR is used (not Tesseract) because it handles printed English well
  and has better confidence scoring.
- PaddleOCR 3.x API (>= 3.0) changed significantly from 2.x:
    - Constructor: use_gpu / show_log / use_angle_cls removed
    - use_angle_cls → use_textline_orientation
    - ocr(arr, cls=True) → predict(arr)   (result format also changed)
  This module handles both versions transparently.
- Results are sorted top-to-bottom, left-to-right.
"""

from __future__ import annotations
import os
import re
import logging
from typing import List, Dict, Any, Optional

# Disable OneDNN (MKL-DNN) in both PaddlePaddle and PaddleX before any paddle imports
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_ENABLE_MKLDNN"] = "0"

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy-load PaddleOCR to avoid slow import at startup
_paddle_ocr = None


def _get_paddle():
    global _paddle_ocr
    if _paddle_ocr is None:
        import os
        # PaddlePaddle + Intel OneDNN has a bug on Windows with certain models:
        # "ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]"
        # Disable OneDNN (MKL-DNN) in both PaddlePaddle and PaddleX to fall back to standard executor.
        os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["PADDLE_ENABLE_MKLDNN"] = "0"

        try:
            from paddleocr import PaddleOCR  # type: ignore[import-untyped]
        except ImportError:
            import sys, pathlib
            venv_site = pathlib.Path(__file__).resolve().parents[3] / ".venv" / "Lib" / "site-packages"
            if venv_site.exists() and str(venv_site) not in sys.path:
                sys.path.insert(0, str(venv_site))
            from paddleocr import PaddleOCR  # type: ignore[import-untyped]
        # PaddleOCR 3.x completely redesigned its constructor:
        #   - use_gpu / show_log removed
        #   - use_angle_cls → use_textline_orientation
        #   - use_doc_orientation_classify / use_doc_unwarping are new optional models
        #     that trigger the OneDNN bug — disable them.
        #   - text_det_unclip_ratio=1.4 and text_det_box_thresh=0.5 prevent DBNet from
        #     missing tight lines or dropping faint text.
        for kwargs in [
            dict(                          # 3.x — tuned det + rec only, disable mkldnn
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_unclip_ratio=1.4,
                text_det_box_thresh=0.5,
                enable_mkldnn=False,
            ),
            dict(                          # 3.x without enable_mkldnn kwarg
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_unclip_ratio=1.4,
                text_det_box_thresh=0.5,
            ),
            dict(                          # 3.x fallback
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            ),
            dict(lang="en", use_angle_cls=True),   # 2.x
            dict(lang="en"),                        # bare minimum
        ]:
            try:
                _paddle_ocr = PaddleOCR(**kwargs)
                logger.info(f"PaddleOCR initialised with kwargs={list(kwargs.keys())}")
                break
            except TypeError as exc:
                logger.warning(f"PaddleOCR init attempt failed {list(kwargs.keys())}: {exc}")
                _paddle_ocr = None
        if _paddle_ocr is None:
            raise RuntimeError("Could not initialise PaddleOCR — check installation.")
    return _paddle_ocr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ocr_image(img: Image.Image) -> Dict[str, Any]:
    """
    Run OCR on a PIL grayscale (or RGB) image.

    Returns:
    {
        "text":        str,          # Full concatenated text in logical reading order
        "lines":       List[dict],   # [{text, confidence, bbox}, ...]
        "confidence":  float,        # Average confidence across all lines
        "ocr_used":    bool,
    }
    """
    arr = _pil_to_np(img)

    lines:       List[dict]  = []
    confidences: List[float] = []

    try:
        ocr = _get_paddle()
    except Exception as e:
        logger.error(f"PaddleOCR init failed: {e}")
        return _empty_result()

    parsed = False
    all_raw_boxes: List[dict] = []
    collected_polys: List[Any] = []

    # --- Try 3.x predict() ---
    try:
        page_results = list(ocr.predict(arr))
        for page_res in page_results:
            json_data = page_res.json if hasattr(page_res, "json") else {}
            res_obj = json_data.get("res") if isinstance(json_data, dict) else None

            # Case A: res_obj is a dict of parallel lists (PaddleX / PaddleOCR 3.7+)
            if isinstance(res_obj, dict):
                rec_texts = res_obj.get("rec_texts") or []
                rec_scores = res_obj.get("rec_scores") or []
                rec_polys = res_obj.get("rec_polys") or res_obj.get("dt_polys") or []
                rec_boxes = res_obj.get("rec_boxes") or []
                collected_polys.extend(rec_polys)

                for idx, text_item in enumerate(rec_texts):
                    text = str(text_item).strip()
                    if not text:
                        continue
                    conf = float(rec_scores[idx]) if idx < len(rec_scores) else 0.9
                    poly = rec_polys[idx] if idx < len(rec_polys) else None
                    box = rec_boxes[idx] if idx < len(rec_boxes) else None

                    if box and len(box) >= 4:
                        x_lft = float(box[0])
                        x_rgt = float(box[2])
                        y_top = float(box[1])
                        y_bot = float(box[3])
                        y_c = (y_top + y_bot) / 2.0
                    elif poly:
                        x_lft = min(pt[0] for pt in poly)
                        x_rgt = max(pt[0] for pt in poly)
                        y_top = min(pt[1] for pt in poly)
                        y_bot = max(pt[1] for pt in poly)
                        y_c = _bbox_y_center(poly)
                    else:
                        x_lft = 0.0
                        x_rgt = 100.0
                        y_top = float(idx * 20)
                        y_bot = float(idx * 20 + 20)
                        y_c = float(idx * 20 + 10)

                    all_raw_boxes.append({
                        "text": text,
                        "confidence": conf,
                        "bbox": poly or box,
                        "x_l": x_lft,
                        "x_r": x_rgt,
                        "w": max(x_rgt - x_lft, 1.0),
                        "y_t": y_top,
                        "y_b": y_bot,
                        "y_c": y_c,
                        "h": max(y_bot - y_top, 1.0),
                    })
                    confidences.append(conf)

            # Case B: res_obj is a list of dicts
            elif isinstance(res_obj, list):
                for item in res_obj:
                    if not isinstance(item, dict):
                        continue
                    text = str(item.get("text", "")).strip()
                    conf = float(item.get("rec_score", item.get("score", 0.0)))
                    raw_poly = item.get("dt_polys") or item.get("rec_polys") or item.get("bbox") or []
                    if raw_poly and not isinstance(raw_poly[0], (list, tuple)):
                        raw_poly = [[raw_poly[i], raw_poly[i + 1]]
                                    for i in range(0, len(raw_poly) - 1, 2)]
                    if not text:
                        continue
                    collected_polys.append(raw_poly)
                    x_lft = min(pt[0] for pt in raw_poly) if raw_poly else 0.0
                    x_rgt = max(pt[0] for pt in raw_poly) if raw_poly else 100.0
                    y_top = min(pt[1] for pt in raw_poly) if raw_poly else 0.0
                    y_bot = max(pt[1] for pt in raw_poly) if raw_poly else 20.0
                    y_c = _bbox_y_center(raw_poly) if raw_poly else 0.0

                    all_raw_boxes.append({
                        "text": text,
                        "confidence": conf,
                        "bbox": raw_poly,
                        "x_l": x_lft,
                        "x_r": x_rgt,
                        "w": max(x_rgt - x_lft, 1.0),
                        "y_t": y_top,
                        "y_b": y_bot,
                        "y_c": y_c,
                        "h": max(y_bot - y_top, 1.0),
                    })
                    confidences.append(conf)

        if all_raw_boxes:
            parsed = True
            logger.debug(f"3.x predict() successfully extracted {len(all_raw_boxes)} boxes")

    except Exception as e_3x:
        logger.warning(f"PaddleOCR 3.x predict() failed ({e_3x}); trying 2.x ocr() API")

    # --- Fall back to 2.x ocr() ---
    if not parsed and hasattr(ocr, "ocr"):
        try:
            try:
                raw = ocr.ocr(arr, cls=True)
            except TypeError:
                raw = ocr.ocr(arr)
            if raw and raw[0] is not None:
                for item in raw[0]:
                    if item is None:
                        continue
                    bbox, text_conf = item
                    text = str(text_conf[0]).strip()
                    conf = float(text_conf[1])
                    if not text:
                        continue
                    x_lft = min(pt[0] for pt in bbox)
                    x_rgt = max(pt[0] for pt in bbox)
                    y_top = min(pt[1] for pt in bbox)
                    y_bot = max(pt[1] for pt in bbox)
                    y_c   = _bbox_y_center(bbox)
                    collected_polys.append(bbox)
                    all_raw_boxes.append({
                        "text": text,
                        "confidence": conf,
                        "bbox": bbox,
                        "x_l": x_lft,
                        "x_r": x_rgt,
                        "w": max(x_rgt - x_lft, 1.0),
                        "y_t": y_top,
                        "y_b": y_bot,
                        "y_c": y_c,
                        "h": max(y_bot - y_top, 1.0),
                    })
                    confidences.append(conf)
            logger.debug(f"2.x ocr() returned {len(all_raw_boxes)} boxes")
        except Exception as e_2x:
            logger.error(f"PaddleOCR 2.x ocr() also failed: {e_2x}")
            return _empty_result()

    if not all_raw_boxes:
        return _empty_result()

    # ------------------------------------------------------------------
    # Step 1: Detect median document skew angle from wide text polygons
    # ------------------------------------------------------------------
    angles = []
    for poly in collected_polys:
        if poly and len(poly) >= 2 and isinstance(poly[0], (list, tuple)) and isinstance(poly[1], (list, tuple)):
            dx = float(poly[1][0] - poly[0][0])
            dy = float(poly[1][1] - poly[0][1])
            if dx > 120:
                deg = float(np.degrees(np.arctan2(dy, dx)))
                if -15.0 <= deg <= 15.0:
                    angles.append(deg)

    med_angle = float(np.median(angles)) if angles else 0.0
    tan_angle = float(np.tan(np.radians(med_angle)))

    # Step 2: Project coordinates along skew angle
    for item in all_raw_boxes:
        x_ref = item["x_l"]
        item["y_t_proj"] = item["y_t"] - x_ref * tan_angle
        item["y_b_proj"] = item["y_b"] - x_ref * tan_angle
        item["y_c_proj"] = (item["y_t_proj"] + item["y_b_proj"]) / 2.0

    # Step 3: Cluster items into logical reading rows
    all_raw_boxes.sort(key=lambda l: l["y_c_proj"])
    rows: List[List[dict]] = []

    for l in all_raw_boxes:
        placed = False
        for r in rows:
            # Check horizontal collision with any item already in this row
            horiz_clash = any(
                (min(l["x_r"], it["x_r"]) - max(l["x_l"], it["x_l"])) / max(min(l["w"], it["w"]), 1.0) > 0.20
                for it in r
            )
            if horiz_clash:
                continue

            # Standalone headers like PART-A, PART B should stay on their own row
            is_part_header = bool(
                re.match(r'^(?:part|section|group)\s*[-:]?\s*[a-z0-9]+', l["text"], re.I) or
                any(re.match(r'^(?:part|section|group)\s*[-:]?\s*[a-z0-9]+', it["text"], re.I) for it in r)
            )
            if is_part_header:
                continue

            # Check vertical overlap with current row
            row_yt = np.mean([it["y_t_proj"] for it in r])
            row_yb = np.mean([it["y_b_proj"] for it in r])
            row_h = np.mean([it["h"] for it in r])

            overlap = min(l["y_b_proj"], row_yb) - max(l["y_t_proj"], row_yt)
            overlap_ratio = overlap / max(min(l["h"], row_h), 1.0)

            if overlap_ratio >= 0.35:
                r.append(l)
                placed = True
                break

        if not placed:
            rows.append([l])

    # Step 4: Sort rows top-to-bottom, and items in each row left-to-right
    rows.sort(key=lambda r: float(np.mean([it["y_c_proj"] for it in r])))

    sorted_lines: List[str] = []
    lines = []
    for r in rows:
        r.sort(key=lambda it: it["x_l"])
        row_text = " ".join(it["text"] for it in r).strip()
        if row_text:
            sorted_lines.append(row_text)
            for it in r:
                lines.append({
                    "text": it["text"],
                    "confidence": it["confidence"],
                    "bbox": it["bbox"],
                    "y_center": it["y_c"],
                    "x_left": it["x_l"],
                })

    avg_conf  = float(np.mean(confidences))
    full_text = "\n".join(sorted_lines)

    return {
        "text":       full_text,
        "lines":      lines,
        "confidence": avg_conf,
        "ocr_used":   True,
    }


def extract_text_native(native_text: str) -> Dict[str, Any]:
    """
    Wrap native (digital) PDF text in the same result format.
    Confidence is set high since it's native text.
    """
    lines = [
        {"text": line.strip(), "confidence": 0.99, "bbox": None,
         "y_center": i, "x_left": 0}
        for i, line in enumerate(native_text.split("\n"))
        if line.strip()
    ]
    return {
        "text":       native_text,
        "lines":      lines,
        "confidence": 0.99,
        "ocr_used":   False,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _pil_to_np(img: Image.Image) -> np.ndarray:
    if img.mode == "L":
        img = img.convert("RGB")
    return np.array(img)


def _empty_result() -> Dict[str, Any]:
    return {"text": "", "lines": [], "confidence": 0.0, "ocr_used": True}


def _bbox_y_center(bbox) -> float:
    ys = [pt[1] for pt in bbox]
    return (min(ys) + max(ys)) / 2
