"""
Image preprocessing pipeline.

Operations applied in order:
1. Deskew (detect rotation, correct it)
2. Denoise (Gaussian blur + threshold)
3. Contrast enhancement (CLAHE)
4. Resolution normalisation (ensure ≥ 200 DPI equivalent)
5. Binarisation (Otsu threshold)

Returns a preprocessed PIL Image (grayscale) suitable for OCR.
"""

import math
import numpy as np
import cv2
from PIL import Image


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(img: Image.Image) -> Image.Image:
    """
    Preprocess image for deep-learning OCR (PaddleOCR).
    Preserves continuous gradients required by DBNet text detector
    while ensuring sufficient resolution for small text and enhancing local line contrast.
    """
    # 1. Ensure minimum resolution for phone captures
    w, h = img.size
    min_dim = min(w, h)
    if min_dim < 1500:
        scale = 1500 / min_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 2. Local contrast enhancement via CLAHE on luminance channel
    # Enhances inter-line separation gutters to prevent DBNet line coalescence
    if img.mode != "RGB":
        img = img.convert("RGB")
    cv_img = np.array(img)
    lab = cv2.cvtColor(cv_img, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_chan)
    lab_enhanced = cv2.merge((l_enhanced, a_chan, b_chan))
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # 3. Suppress colored student pen scribbles, ticks, and ink notes
    # Typeset examination text is neutral black/charcoal; annotations are colored ballpoint/gel ink
    hsv = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)
    blue_ink = (h_chan >= 85) & (h_chan <= 145) & (s_chan >= 30) & (v_chan >= 40)
    red_ink = ((h_chan <= 15) | (h_chan >= 165)) & (s_chan >= 35) & (v_chan >= 40)
    green_ink = (h_chan >= 35) & (h_chan <= 85) & (s_chan >= 35) & (v_chan >= 40)
    colored_ink_mask = blue_ink | red_ink | green_ink

    if np.any(colored_ink_mask):
        clean_rgb = enhanced_rgb.copy()
        clean_rgb[colored_ink_mask] = [255, 255, 255]
        return Image.fromarray(clean_rgb)

    return Image.fromarray(enhanced_rgb)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _pil_to_cv(img: Image.Image) -> np.ndarray:
    if img.mode != "L":
        img = img.convert("L")
    return np.array(img)


def _cv_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))


def _denoise(arr: np.ndarray) -> np.ndarray:
    """Apply a mild Gaussian blur to reduce scan noise."""
    return cv2.GaussianBlur(arr, (3, 3), 0)


def _deskew(arr: np.ndarray) -> np.ndarray:
    """
    Detect dominant text skew angle and rotate to correct it.
    Uses the Hough line transform on a binarised copy.
    Only corrects if skew is between -15° and 15° (to avoid false corrections).
    """
    # Binarise for skew detection
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find edges
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    # Hough lines
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=100,
                             minLineLength=arr.shape[1] // 4, maxLineGap=20)
    if lines is None:
        return arr

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if -15 < angle < 15:
            angles.append(angle)

    if not angles:
        return arr

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return arr  # Not worth correcting

    (h, w) = arr.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(arr, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def _enhance_contrast(arr: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation).
    Helps with uneven scan lighting.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(arr)


def _binarise(arr: np.ndarray) -> np.ndarray:
    """
    Adaptive thresholding — better than global Otsu for varying lighting.
    """
    binary = cv2.adaptiveThreshold(
        arr, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10
    )
    return binary
