import statistics
import sys
import time
from pathlib import Path
from PIL import Image

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.services.embeddings.image_service import (
    ImageEmbeddingService,
    get_image_embedding_service,
)
from app.services.embeddings.text_service import (
    TextEmbeddingService,
    get_text_embedding_service,
)


def run_benchmark(iterations: int = 10):
    print("\n========================================================")
    print("      OUTLET VERIFICATION - EMBEDDING BENCHMARK (CPU)")
    print("========================================================\n")

    # 1. Text Model Load Time
    print("--- 1. Text Model ('all-MiniLM-L6-v2') ---")
    text_service = get_text_embedding_service()
    if not text_service._is_loaded:
        t0 = time.perf_counter()
        text_service.load_model()
        text_load_time = time.perf_counter() - t0
    else:
        text_load_time = text_service.load_duration_seconds
    print(f"Model Load Time:    {text_load_time:.3f} s")

    # 2. Text Inference Latency
    sample_text = "Apollo Pharmacy - Indiranagar 100ft Road, Bangalore"
    # Warmup
    text_service.generate_name_embedding(sample_text)

    text_latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        emb = text_service.generate_name_embedding(sample_text)
        text_latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

    avg_text = statistics.mean(text_latencies)
    median_text = statistics.median(text_latencies)
    min_text = min(text_latencies)
    max_text = max(text_latencies)

    print(f"Iterations:         {iterations}")
    print(f"Vector Dimensions:  {len(emb)}")
    print(f"Average Latency:    {avg_text:.2f} ms")
    print(f"Median (p50):       {median_text:.2f} ms")
    print(f"Min / Max:          {min_text:.2f} ms / {max_text:.2f} ms")

    # 3. Image Model Load Time
    print("\n--- 2. Image Model ('CLIP ViT-B/32') ---")
    image_service = get_image_embedding_service()
    if not image_service._is_loaded:
        t0 = time.perf_counter()
        image_service.load_model()
        image_load_time = time.perf_counter() - t0
    else:
        image_load_time = image_service.load_duration_seconds
    print(f"Model Load Time:    {image_load_time:.3f} s")

    # 4. Image Inference Latency
    sample_image = Image.new("RGB", (224, 224), color=(73, 109, 137))
    # Warmup
    image_service.generate_image_embedding(sample_image)

    image_latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        img_emb = image_service.generate_image_embedding(sample_image)
        image_latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

    avg_img = statistics.mean(image_latencies)
    median_img = statistics.median(image_latencies)
    min_img = min(image_latencies)
    max_img = max(image_latencies)

    print(f"Iterations:         {iterations}")
    print(f"Vector Dimensions:  {len(img_emb)}")
    print(f"Average Latency:    {avg_img:.2f} ms")
    print(f"Median (p50):       {median_img:.2f} ms")
    print(f"Min / Max:          {min_img:.2f} ms / {max_img:.2f} ms")

    print("\n--------------------------------------------------------")
    print("                 BENCHMARK SUMMARY                      ")
    print("--------------------------------------------------------")
    print(f"Text Model Load:     {text_load_time:.3f} s")
    print(f"Image Model Load:    {image_load_time:.3f} s")
    print(f"Text Inference avg:  {avg_text:.2f} ms  (dim: {len(emb)})")
    print(f"Image Inference avg: {avg_img:.2f} ms (dim: {len(img_emb)})")
    print("--------------------------------------------------------\n")

    return {
        "text_load_time": text_load_time,
        "image_load_time": image_load_time,
        "text_latency_avg_ms": avg_text,
        "text_latency_median_ms": median_text,
        "image_latency_avg_ms": avg_img,
        "image_latency_median_ms": median_img,
        "text_dimensions": len(emb),
        "image_dimensions": len(img_emb),
    }


if __name__ == "__main__":
    run_benchmark(iterations=10)
