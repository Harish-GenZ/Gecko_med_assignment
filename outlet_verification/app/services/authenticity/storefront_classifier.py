import io
import logging
import threading
from typing import Any, NamedTuple
import torch

from app.services.authenticity.utils import load_pil_image

logger = logging.getLogger("outlet_verification.authenticity.visual")

MODEL_NAME = "openai/clip-vit-base-patch32"

# Domain-specific candidate labels for zero-shot retail outlet classification
CANDIDATE_LABELS = [
    "a storefront photograph of a pharmacy, medical store, chemist shop, supermarket, or retail grocery shop",
    "an exterior view of a retail department store, medical shop, pharmacy, or commercial storefront with signboard",
    "a selfie or portrait photograph of a person",
    "a photo of an empty street, road, or outdoor landscape",
    "an indoor living room, bedroom, office desk, table, or wall in a house",
    "a photo of a vehicle, car, or motorcycle",
    "a blank, solid black, solid white, or completely blurry picture",
    "a receipt, bill document, or paper text screenshot",
]

RETAIL_INDICES = {0, 1}


class VisualClassificationResult(NamedTuple):
    is_retail_store: bool
    store_probability: float
    top_predicted_label: str
    rejection_reason: str | None


class StorefrontClassifier:
    """
    Zero-shot multimodal classifier using CLIP ViT-B/32.
    Evaluates whether an uploaded image is authentically a retail store, supermarket,
    or general store storefront vs an invalid/fake input (selfie, empty road, living room,
    vehicle, blank/blurry image, document/receipt).
    """
    _instance: "StorefrontClassifier | None" = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self.processor: Any = None
        self.model: Any = None
        self.device: str = "cpu"
        self._is_loaded: bool = False

    @classmethod
    def get_instance(cls) -> "StorefrontClassifier":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load_model(self) -> None:
        if self._is_loaded:
            return

        with self._lock:
            if self._is_loaded:
                return

            logger.info("Initializing CLIP zero-shot storefront classifier '%s' on %s...", MODEL_NAME, self.device)
            from transformers import CLIPModel, CLIPProcessor

            self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
            self.model = CLIPModel.from_pretrained(MODEL_NAME).to(self.device)
            self.model.eval()
            self._is_loaded = True
            logger.info("CLIP zero-shot storefront classifier successfully loaded.")

    def classify_storefront(self, image_input: Any, min_store_prob: float = 0.35) -> VisualClassificationResult:
        """
        Classifies an input image against retail store vs invalid non-store categories.
        """
        self.load_model()
        pil_img = load_pil_image(image_input)

        inputs = self.processor(
            text=CANDIDATE_LABELS,
            images=pil_img,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0].cpu().numpy()

        store_prob = float(sum(probs[idx] for idx in RETAIL_INDICES))
        top_idx = int(probs.argmax())
        top_label = CANDIDATE_LABELS[top_idx]

        is_store = (top_idx in RETAIL_INDICES) or (store_prob >= min_store_prob)

        rejection_reason: str | None = None
        if not is_store:
            rejection_reason = (
                f"Image does not appear to be an authentic commercial store or supermarket "
                f"(top prediction: '{top_label}', retail store probability: {store_prob:.2%})."
            )

        return VisualClassificationResult(
            is_retail_store=is_store,
            store_probability=round(store_prob, 4),
            top_predicted_label=top_label,
            rejection_reason=rejection_reason,
        )
