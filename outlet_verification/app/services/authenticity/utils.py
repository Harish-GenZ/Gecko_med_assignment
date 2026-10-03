import io
from pathlib import Path
from typing import Any
from PIL import Image


def load_pil_image(image_input: Any) -> Image.Image:
    """
    Parses and validates various image inputs into a valid RGB PIL Image.
    Supports PIL Image, bytes, BytesIO, or file path.
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
            raise ValueError(f"Unsupported image input type: {type(image_input).__name__}")

        # Ensure image is loaded and converted to RGB
        img.load()
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Failed to load image: {exc}") from exc
