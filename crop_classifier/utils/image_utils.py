"""
utils/image_utils.py
────────────────────
Preprocessing and augmentation helpers used both in training and in the Streamlit UI.
"""

from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image


IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_and_resize(image_source, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load an image from a file path or a file-like object, resize it,
    and return a uint8 numpy array of shape (H, W, 3).
    """
    if isinstance(image_source, (str, bytes)):
        img = Image.open(image_source)
    else:
        img = Image.open(image_source)
    return np.array(img.convert("RGB").resize(target_size), dtype=np.uint8)


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalise a uint8 or float32 image to [0, 1] then apply
    ImageNet mean/std normalisation.
    """
    img = image.astype(np.float32) / 255.0
    img = (img - IMG_MEAN) / IMG_STD
    return img


def preprocess_for_inference(
    image_source,
    target_size: Tuple[int, int] = (224, 224),
    use_imagenet_norm: bool = False,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single image.

    Returns a float32 array of shape (1, H, W, 3) ready for model.predict().
    """
    img = load_and_resize(image_source, target_size)
    if use_imagenet_norm:
        arr = normalize(img)
    else:
        arr = img.astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def pil_to_bytes(image: Image.Image, fmt: str = "JPEG") -> bytes:
    """Convert a PIL image to raw bytes (useful for Streamlit download buttons)."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def bytes_to_pil(raw: bytes) -> Image.Image:
    return Image.open(io.BytesIO(raw)).convert("RGB")


def overlay_prediction(
    image: Image.Image,
    label: str,
    confidence: float,
    font_size: int = 20,
) -> Image.Image:
    """
    Draw a prediction label + confidence bar on a copy of the image.
    Returns a new PIL Image (does not modify the original).
    """
    from PIL import ImageDraw, ImageFont

    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)

    # Background rect for text
    bar_height = 40
    draw.rectangle([0, 0, out.width, bar_height], fill=(0, 0, 0, 180))

    text = f"{label}  {confidence * 100:.1f}%"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    draw.text((8, 8), text, fill="white", font=font)
    return out
