import logging
import threading
import time
from typing import Any

from app.services.embeddings.base import normalize_vector, validate_name_text

logger = logging.getLogger("outlet_verification.embeddings.text")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIMENSION = 384


class TextEmbeddingService:
    """
    Singleton service managing the 'all-MiniLM-L6-v2' text embedding model.
    Generates 384-dimensional normalized embeddings for outlet names.
    Loaded once per application worker on CPU and reused.
    """
    _instance: "TextEmbeddingService | None" = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self.model: Any = None
        self.device: str = "cpu"
        self._is_loaded: bool = False
        self.load_duration_seconds: float = 0.0

    @classmethod
    def get_instance(cls) -> "TextEmbeddingService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load_model(self) -> None:
        """
        Loads the SentenceTransformer model once on CPU.
        """
        if self._is_loaded:
            return

        with self._lock:
            if self._is_loaded:
                return

            logger.info("Initializing text embedding model '%s' on CPU...", MODEL_NAME)
            start_time = time.perf_counter()

            try:
                # Prefer sentence_transformers if installed
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(MODEL_NAME, device=self.device)
            except ImportError:
                # Fallback to pure transformers AutoModel + AutoTokenizer
                logger.info("sentence_transformers not found; falling back to transformers AutoModel...")
                from transformers import AutoModel, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
                self.model.eval()

            self.load_duration_seconds = time.perf_counter() - start_time
            self._is_loaded = True
            logger.info(
                "Text embedding model loaded successfully in %.2f seconds (device: %s).",
                self.load_duration_seconds,
                self.device,
            )

    def generate_name_embedding(self, name: str) -> list[float]:
        """
        Generates a 384-dimensional, L2-normalized embedding vector for an outlet name.
        """
        # 1. Input validation
        cleaned_name = validate_name_text(name)

        # 2. Ensure model is loaded
        if not self._is_loaded:
            self.load_model()

        # 3. Model inference
        try:
            if hasattr(self.model, "encode"):
                # SentenceTransformer path
                embedding = self.model.encode(
                    cleaned_name,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                vector = [float(x) for x in embedding.tolist()]
            else:
                # Transformers AutoModel path
                import torch
                inputs = self.tokenizer(
                    cleaned_name,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Mean pooling with attention mask
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)
                    token_embeddings = outputs.last_hidden_state
                    sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
                    sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                    pooled = sum_embeddings / sum_mask
                    normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    vector = [float(x) for x in normalized[0].cpu().numpy().tolist()]

        except Exception as exc:
            logger.error("Error during text embedding inference for '%s': %s", cleaned_name, exc)
            raise RuntimeError(f"Text embedding generation failed: {exc}") from exc

        # 4. Dimension & normalization check
        if len(vector) != EXPECTED_DIMENSION:
            raise ValueError(
                f"Generated embedding dimension {len(vector)} does not match expected {EXPECTED_DIMENSION}."
            )

        # Explicitly ensure normalized vector
        return normalize_vector(vector)


def get_text_embedding_service() -> TextEmbeddingService:
    """Convenience getter for singleton TextEmbeddingService."""
    return TextEmbeddingService.get_instance()


def generate_name_embedding(name: str) -> list[float]:
    """Top-level function generating 384-dim normalized text embedding."""
    service = get_text_embedding_service()
    return service.generate_name_embedding(name)
