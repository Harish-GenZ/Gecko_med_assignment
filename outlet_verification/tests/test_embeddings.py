import io
import math
import sys
from contextlib import contextmanager
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from PIL import Image

from app.db.session import SessionLocal
from app.models.outlet import Outlet
from app.services.embeddings import (
    cosine_similarity,
    generate_image_embedding,
    generate_name_embedding,
    normalize_vector,
)


@contextmanager
def assert_raises(exc_type, match=None):
    """Simple standard-library context manager replacing pytest.raises."""
    try:
        yield
    except exc_type as exc:
        if match and match not in str(exc):
            raise AssertionError(f"Expected error message containing '{match}', got '{exc}'")
    except Exception as exc:
        raise AssertionError(f"Expected exception {exc_type.__name__}, got {type(exc).__name__}: {exc}")
    else:
        raise AssertionError(f"Expected exception {exc_type.__name__} was not raised.")


def create_dummy_image(color=(73, 109, 137), size=(224, 224)) -> Image.Image:
    """Helper to generate a clean synthetic PIL test image."""
    return Image.new("RGB", size, color=color)


def create_dummy_image_bytes(color=(255, 100, 50)) -> bytes:
    """Helper to generate encoded JPEG bytes."""
    img = create_dummy_image(color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ==============================================================================
# 1. TEXT EMBEDDING TESTS
# ==============================================================================

def test_text_embedding_valid():
    name = "Apollo Pharmacy - Indiranagar 100ft Road"
    embedding = generate_name_embedding(name)

    assert isinstance(embedding, list), "Embedding should be a list"
    assert len(embedding) == 384, f"Expected 384 dimensions, got {len(embedding)}"
    assert all(isinstance(x, (float, int)) for x in embedding), "All items must be numeric"

    # Verify L2 norm is ~1.0
    norm = math.sqrt(sum(x ** 2 for x in embedding))
    assert abs(norm - 1.0) < 1e-4, f"Vector should be unit normalized, got norm={norm}"


def test_text_embedding_empty_or_whitespace():
    with assert_raises(ValueError, match="cannot be empty or whitespace-only"):
        generate_name_embedding("")

    with assert_raises(ValueError, match="cannot be empty or whitespace-only"):
        generate_name_embedding("     \n\t   ")


def test_text_embedding_invalid_type():
    with assert_raises(ValueError, match="cannot be None"):
        generate_name_embedding(None)

    with assert_raises(ValueError, match="must be a string"):
        generate_name_embedding(12345)


# ==============================================================================
# 2. IMAGE EMBEDDING TESTS
# ==============================================================================

def test_image_embedding_valid_pil():
    img = create_dummy_image()
    embedding = generate_image_embedding(img)

    assert isinstance(embedding, list), "Embedding should be a list"
    assert len(embedding) == 512, f"Expected 512 dimensions, got {len(embedding)}"
    assert all(isinstance(x, (float, int)) for x in embedding), "All items must be numeric"

    # Verify L2 norm is ~1.0
    norm = math.sqrt(sum(x ** 2 for x in embedding))
    assert abs(norm - 1.0) < 1e-4, f"Vector should be unit normalized, got norm={norm}"


def test_image_embedding_valid_bytes():
    img_bytes = create_dummy_image_bytes()
    embedding = generate_image_embedding(img_bytes)

    assert isinstance(embedding, list)
    assert len(embedding) == 512
    norm = math.sqrt(sum(x ** 2 for x in embedding))
    assert abs(norm - 1.0) < 1e-4


def test_image_embedding_corrupt_or_empty():
    with assert_raises(ValueError, match="cannot be empty"):
        generate_image_embedding(b"")

    with assert_raises(ValueError, match="Corrupt or invalid image"):
        generate_image_embedding(b"NOT_A_VALID_IMAGE_CONTENT_XYZ")

    with assert_raises(ValueError, match="cannot be None"):
        generate_image_embedding(None)


# ==============================================================================
# 3. EMBEDDING QUALITY / SIMILARITY SANITY CHECKS
# ==============================================================================

def test_cosine_similarity_computation():
    emb1 = generate_name_embedding("ABC Medicals")
    emb2 = generate_name_embedding("ABC Medical Store")

    sim = cosine_similarity(emb1, emb2)
    assert isinstance(sim, float)
    assert -1.0 <= sim <= 1.0
    # Sanity check: similar names should have positive similarity
    assert sim > 0.5, f"Expected positive similarity for related names, got {sim}"


def test_image_similarity_computation():
    img1 = create_dummy_image(color=(200, 50, 50))
    img2 = create_dummy_image(color=(205, 55, 52))  # Very close color
    img3 = create_dummy_image(color=(10, 200, 30))   # Different color

    emb1 = generate_image_embedding(img1)
    emb2 = generate_image_embedding(img2)
    emb3 = generate_image_embedding(img3)

    sim_close = cosine_similarity(emb1, emb2)
    sim_diff = cosine_similarity(emb1, emb3)

    assert -1.0 <= sim_close <= 1.0
    assert -1.0 <= sim_diff <= 1.0
    # Close images should have higher similarity than completely different images
    assert sim_close > sim_diff, f"Expected {sim_close} > {sim_diff}"


# ==============================================================================
# 4. DATABASE VECTOR INTEGRATION TESTS
# ==============================================================================

def test_database_vector_persistence():
    db = SessionLocal()
    outlet_with_embeddings_id = None
    outlet_null_embeddings_id = None

    try:
        # A. Insert an outlet with both 384-dim name_embedding and 512-dim image_embedding
        name = "MedPlus Health Services - Test Branch"
        text_emb = generate_name_embedding(name)
        img_emb = generate_image_embedding(create_dummy_image())

        outlet = Outlet(
            name=name,
            latitude=12.9352,
            longitude=77.6245,
            image_url="https://example.com/medplus.jpg",
            name_embedding=text_emb,
            image_embedding=img_emb,
        )
        db.add(outlet)
        db.commit()
        db.refresh(outlet)
        outlet_with_embeddings_id = outlet.id

        # Verify retrieval
        saved = db.query(Outlet).filter(Outlet.id == outlet_with_embeddings_id).first()
        assert saved is not None
        assert saved.name == name
        assert len(saved.name_embedding) == 384, f"Expected 384, got {len(saved.name_embedding)}"
        assert len(saved.image_embedding) == 512, f"Expected 512, got {len(saved.image_embedding)}"

        # B. Insert an outlet with NULL embeddings (verifies backwards compatibility)
        outlet_null = Outlet(
            name="Legacy Unembedded Outlet",
            latitude=13.0827,
            longitude=80.2707,
            image_url="https://example.com/legacy.jpg",
            name_embedding=None,
            image_embedding=None,
        )
        db.add(outlet_null)
        db.commit()
        db.refresh(outlet_null)
        outlet_null_embeddings_id = outlet_null.id

        saved_null = db.query(Outlet).filter(Outlet.id == outlet_null_embeddings_id).first()
        assert saved_null is not None
        assert saved_null.name_embedding is None
        assert saved_null.image_embedding is None

    finally:
        # Clean up test rows
        if outlet_with_embeddings_id:
            db.query(Outlet).filter(Outlet.id == outlet_with_embeddings_id).delete()
        if outlet_null_embeddings_id:
            db.query(Outlet).filter(Outlet.id == outlet_null_embeddings_id).delete()
        db.commit()
        db.close()


if __name__ == "__main__":
    print("\n========================================================")
    print("      RUNNING PHASE 3B EMBEDDING & DATABASE TESTS")
    print("========================================================\n")
    test_text_embedding_valid()
    print("[PASS] test_text_embedding_valid (384 dimensions, normalized)")
    test_text_embedding_empty_or_whitespace()
    print("[PASS] test_text_embedding_empty_or_whitespace")
    test_text_embedding_invalid_type()
    print("[PASS] test_text_embedding_invalid_type")
    test_image_embedding_valid_pil()
    print("[PASS] test_image_embedding_valid_pil (512 dimensions, normalized)")
    test_image_embedding_valid_bytes()
    print("[PASS] test_image_embedding_valid_bytes")
    test_image_embedding_corrupt_or_empty()
    print("[PASS] test_image_embedding_corrupt_or_empty")
    test_cosine_similarity_computation()
    print("[PASS] test_cosine_similarity_computation (text similarity)")
    test_image_similarity_computation()
    print("[PASS] test_image_similarity_computation (image similarity)")
    test_database_vector_persistence()
    print("[PASS] test_database_vector_persistence (pgvector 384 & 512, NULL backwards compatibility)")
    print("\n--------------------------------------------------------")
    print("      ALL TESTS PASSED SUCCESSFULLY!")
    print("--------------------------------------------------------\n")
