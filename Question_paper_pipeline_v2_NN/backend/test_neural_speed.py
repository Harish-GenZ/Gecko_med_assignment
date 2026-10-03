import time
from pathlib import Path
from PIL import Image
from app.services.layout.layout_detector import get_layout_detector
from app.services.ocr.ocr_engine import ocr_image
from app.services.vision.local_vlm import verify_page_with_local_vlm
from app.services.classification.question_classifier import classify_question

img_path = Path('../raw_datasets/UHSR/UHSR/WhatsApp Image 2026-09-26 at 06.32.03.jpeg')
if not img_path.exists():
    print(f"File not found: {img_path}")
    exit(1)

img = Image.open(img_path)
print(f"Testing on real image: {img_path.name} (size: {img.size})")

t0 = time.time()
# NN Stage 1: Layout detection
detector = get_layout_detector()
layout = detector.detect_layout(img)
t1 = time.time()
print(f"NN Stage 1 (Layout CNN): {t1-t0:.3f}s | Header: {layout.header_region is not None} | Noise regions: {len(layout.noise_regions)}")

# NN Stage 2: OCR with layout noise masking
ocr_res = ocr_image(img, mask_noise=True)
t2 = time.time()
lines_count = len(ocr_res.get("lines", []))
chars_count = len(ocr_res.get("text", ""))
print(f"NN Stage 2 (RapidOCR ONNX): {t2-t1:.3f}s | Lines: {lines_count} | Chars: {chars_count}")

# NN Stage 3: Local VLM Header Verification
payload = {"metadata": {}, "instructions": [], "questions": []}
vlm_res = verify_page_with_local_vlm(1, img, payload)
t3 = time.time()
print(f"NN Stage 3 (Local VLM Micro-Crop): {t3-t2:.3f}s | Metadata corrections: {len(vlm_res.metadata_corrections)}")
for mc in vlm_res.metadata_corrections:
    print(f"   -> {mc.field}: {mc.value}")

# NN Stage 4: Question Classification
q_sample = {"text": ocr_res["text"][:150], "marks": "10", "number": "1"}
q_type, q_conf = classify_question(q_sample)
t4 = time.time()
print(f"NN Stage 4 (Embedding Classifier): {t4-t3:.3f}s | Question Type: {q_type.value} (conf: {q_conf})")

print(f"\n==========================================")
print(f"TOTAL PIPELINE RUNTIME: {t4-t0:.2f} seconds!")
print(f"Previous architecture took: ~150 seconds per page")
print(f"Speedup: {(150.0 / max(t4-t0, 0.01)):.1f}x faster!")
print(f"==========================================")
