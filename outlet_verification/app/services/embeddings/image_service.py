import io
import logging
import threading
import time
from pathlib import Path
from typing import Any, Union
from PIL import Image, UnidentifiedImageError

from app.services.embeddings.base import normalize_vector

logger = logging.getLogger("outlet_verification.embeddings.image")

MODEL_NAME = "openai/clip-vit-base-patch32"
EXPECTED_DIMENSION = 512

ImageInput = Union[str, Path, bytes, io.BytesIO, Image.Image]


class ImageEmbeddingService:
    """
    Singleton service managing the 'CLIP ViT-B/32' image embedding model.
    Generates 512-dimensional normalized visual embeddings for outlet photographs.
    Loaded once per application worker on CPU and reused.
    """
    _instance: "ImageEmbeddingService | None" = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self.device: str = "cpu"
        self._is_loaded: bool = False
        self.load_duration_seconds: float = 0.0

    @classmethod
    def get_instance(cls) -> "ImageEmbeddingService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load_model(self) -> None:
        """
        Loads the HuggingFace CLIPModel and CLIPProcessor once on CPU.
        """
        if self._is_loaded:
            return

        with self._lock:
            if self._is_loaded:
                return

            logger.info("Initializing image embedding model '%s' on CPU...", MODEL_NAME)
            start_time = time.perf_counter()

            try:
                from transformers import CLIPModel, CLIPProcessor
                self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
                self.model = CLIPModel.from_pretrained(MODEL_NAME).to(self.device)
                self.model.eval()
            except Exception as exc:
                logger.error("Failed to load CLIP model '%s': %s", MODEL_NAME, exc)
                raise RuntimeError(f"Image model loading failed: {exc}") from exc

            self.load_duration_seconds = time.perf_counter() - start_time
            self._is_loaded = True
            logger.info(
                "Image embedding model loaded successfully in %.2f seconds (device: %s).",
                self.load_duration_seconds,
                self.device,
            )

    def _load_pil_image(self, image_input: ImageInput) -> Image.Image:
        """
        Parses and validates various image inputs into a valid RGB PIL Image.
        Raises ValueError for invalid, empty, or corrupt image data.
        """
        if image_input is None:
            raise ValueError("Image input cannot be None.")

        try:
            if isinstance(image_input, Image.Image):
                img = image_input
            elif isinstance(image_input, bytes):
                if len(image_input) == 0:
                    raise ValueError("Image bytes cannot be empty.")
                img = Image.open(io.BytesIO(image_input))
            elif isinstance(image_input, io.BytesIO):
                img = Image.open(image_input)
            elif isinstance(image_input, (str, Path)):
                path_obj = Path(image_input)
                if not path_obj.exists():
                    raise ValueError(f"Image file does not exist at path: {image_input}")
                img = Image.open(path_obj)
            else:
                raise ValueError(
                    f"Unsupported image input type: {type(image_input).__name__}. "
                    "Expected PIL Image, file path, bytes, or BytesIO."
                )

            # Force loading the image data to catch corrupt file errors early
            img.load()
            return img.convert("RGB")

        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Corrupt or invalid image data: {exc}") from exc

    def generate_image_embedding(self, image: ImageInput) -> list[float]:
        """
        Generates a 512-dimensional, L2-normalized embedding vector for an outlet photo.
        """
        # 1. Preprocess and validate image
        pil_image = self._load_pil_image(image)

        # 2. Ensure model is loaded
        if not self._is_loaded:
            self.load_model()

        # 3. Model inference
        try:
            import torch

            # Process image using CLIPProcessor
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                # Extract image features from CLIP vision tower
                image_features = self.model.get_image_features(**inputs)

                # In Transformers 5.x, get_image_features returns BaseModelOutputWithPooling
                if isinstance(image_features, torch.Tensor):
                    raw_tensor = image_features
                elif hasattr(image_features, "pooler_output") and image_features.pooler_output is not None:
                    raw_tensor = image_features.pooler_output
                elif hasattr(image_features, "image_embeds") and image_features.image_embeds is not None:
                    raw_tensor = image_features.image_embeds
                else:
                    raw_tensor = image_features[0]
                
                # Normalize features
                normalized_features = torch.nn.functional.normalize(raw_tensor, p=2, dim=1)
                vector = [float(x) for x in normalized_features[0].cpu().numpy().tolist()]

        except Exception as exc:
            logger.error("Error during image embedding inference: %s", exc)
            raise RuntimeError(f"Image embedding generation failed: {exc}") from exc

        # 4. Dimension & normalization check
        if len(vector) != EXPECTED_DIMENSION:
            raise ValueError(
                f"Generated image embedding dimension {len(vector)} does not match expected {EXPECTED_DIMENSION}."
            )

        # Explicitly ensure normalized vector
        return normalize_vector(vector)


def get_image_embedding_service() -> ImageEmbeddingService:
    """Convenience getter for singleton ImageEmbeddingService."""
    return ImageEmbeddingService.get_instance()


def generate_image_embedding(image: ImageInput) -> list[float]:
    """Top-level function generating 512-dim normalized image embedding."""
    service = get_image_embedding_service()
    return service.generate_image_embedding(image)
