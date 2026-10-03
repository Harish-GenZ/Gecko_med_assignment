"""
Image preprocessing pipeline.

Operations applied in order:
1. Resolution normalisation (ensure adequate DPI for deep-learning OCR)
2. Deskew (detect rotation angle from text baselines and correct it)
3. Adaptive Illumination & Color Normalization for Dim/Shadowy Pages:
   - Dynamic Gamma Lift on underexposed/dim captures
   - Background illumination surface estimation & division (removes shadows & gray haze)
   - Adaptive CLAHE contrast enhancement
   - Auto white-balance and percentile dynamic range expansion
4. Suppress colored student pen scribbles, ticks, and ink notes

Returns an enhanced PIL Image (RGB) with bright white paper and crisp, high-contrast dark text.
"""

import math
import numpy as np
import cv2
from PIL import Image


def preprocess(img: Image.Image) -> Image.Image:
    """
    Preprocess image for neural OCR and vision verification.
    Normalizes dim, shadowy, or discolored document pages into clean, bright, high-contrast images.
    """
    try:
        # 1. Ensure minimum resolution for phone captures (targeting ~1800px minimum dimension)
        w, h = img.size
        min_dim = min(w, h)
        if min_dim < 1800:
            scale = min(2.5, 1800.0 / min_dim)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        if img.mode != "RGB":
            img = img.convert("RGB")
        cv_img = np.array(img)

        # 2. Automated Deskew (straightens tilted mobile captures)
        cv_img = _deskew_image(cv_img)

        # 3. Adaptive Illumination, Color & Contrast Enhancement for Dim Documents
        cv_img = _enhance_illumination_and_color(cv_img)

        # 4. Suppress colored student pen scribbles, ticks, and ink notes
        cv_img = _suppress_colored_annotations(cv_img)

        return Image.fromarray(cv_img)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"[Preprocessor] Preprocessing exception: {e}. Falling back to input image.")
        return img


def _detect_skew_angle(gray: np.ndarray) -> float:
    """Detect dominant text skew angle using Hough line transform on a downscaled binary copy."""
    h, w = gray.shape[:2]
    scale = min(1.0, 900.0 / max(h, w))
    if scale < 1.0:
        small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = gray

    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=80,
                            minLineLength=small.shape[1] // 5, maxLineGap=15)
    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if -15 < angle < 15:
            angles.append(angle)

    if not angles:
        return 0.0

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return 0.0
    return median_angle


def _deskew_image(cv_img: np.ndarray) -> np.ndarray:
    """Rotates image to 0° baseline alignment if skew is detected."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    angle = _detect_skew_angle(gray)
    if abs(angle) < 0.5:
        return cv_img

    (h, w) = cv_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        cv_img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def _enhance_illumination_and_color(cv_img: np.ndarray) -> np.ndarray:
    """
    Normalizes dim, underexposed, or unevenly illuminated pages:
    - Performs adaptive gamma lift on dim captures.
    - Flattens illumination gradients and shadows via background division.
    - Enhances local line contrast via CLAHE.
    - Performs percentile white-balance to lock paper to pure white (245-255).
    """
    lab = cv2.cvtColor(cv_img, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    mean_l = float(np.mean(l_chan))
    p95_l = float(np.percentile(l_chan, 95))
    is_dim = (mean_l < 155 or p95_l < 210)

    # 1. Dynamic Gamma Lift if page is dim or underexposed
    if is_dim:
        target_mid = 0.55
        current_mid = max(mean_l, 15.0) / 255.0
        gamma = float(np.clip(np.log(target_mid) / np.log(current_mid), 0.45, 0.90))
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
        cv_img = cv2.LUT(cv_img, table)
        lab = cv2.cvtColor(cv_img, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)

    # 2. Background Illumination / Shadow Flattening (Downsampled flat-field division)
    # Downsamples to 320px for ultra-fast (16ms) low-memory background estimation without heavy RAM spikes
    orig_h, orig_w = l_chan.shape[:2]
    small_w = 320
    small_h = max(32, int(small_w * orig_h / max(orig_w, 1)))
    small_l = cv2.resize(l_chan, (small_w, small_h), interpolation=cv2.INTER_AREA)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    bg_small = cv2.morphologyEx(small_l, cv2.MORPH_CLOSE, kernel)
    bg_small = cv2.GaussianBlur(bg_small, (21, 21), 0)
    bg_l = cv2.resize(bg_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    bg_l_safe = np.maximum(bg_l, 1).astype(np.float32)
    l_norm = np.clip((l_chan.astype(np.float32) / bg_l_safe) * 235.0, 0, 255).astype(np.uint8)

    # 3. Adaptive CLAHE on normalized luminance channel
    clip_lim = 3.0 if is_dim else 2.0
    clahe = cv2.createCLAHE(clipLimit=clip_lim, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_norm)

    # 4. Recombine and perform White-Balance / Percentile Dynamic Range Stretch
    lab_enhanced = cv2.merge((l_enhanced, a_chan, b_chan))
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    channels = cv2.split(enhanced_rgb)
    out_channels = []
    for ch in channels:
        p_low = np.percentile(ch, 1)
        p_high = np.percentile(ch, 98)
        if p_high > p_low:
            stretched = np.clip((ch.astype(np.float32) - p_low) * (255.0 / (p_high - p_low)), 0, 255)
            out_channels.append(stretched.astype(np.uint8))
        else:
            out_channels.append(ch)

    return cv2.merge(out_channels)


def _suppress_colored_annotations(cv_img: np.ndarray) -> np.ndarray:
    """Suppress colored student pen scribbles, ticks, and ink notes (blue/red/green)."""
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_RGB2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)
    blue_ink = (h_chan >= 85) & (h_chan <= 145) & (s_chan >= 30) & (v_chan >= 40)
    red_ink = ((h_chan <= 15) | (h_chan >= 165)) & (s_chan >= 35) & (v_chan >= 40)
    green_ink = (h_chan >= 35) & (h_chan <= 85) & (s_chan >= 35) & (v_chan >= 40)
    colored_ink_mask = blue_ink | red_ink | green_ink

    if np.any(colored_ink_mask):
        clean_rgb = cv_img.copy()
        clean_rgb[colored_ink_mask] = [255, 255, 255]
        return clean_rgb
    return cv_img
