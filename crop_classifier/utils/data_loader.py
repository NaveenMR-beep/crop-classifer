"""
utils/data_loader.py
────────────────────
Builds TensorFlow tf.data.Dataset pipelines from the crop CSV + image directories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


CLASSES = ["jute", "maize", "rice", "sugarcane", "wheat"]
NUM_CLASSES = len(CLASSES)
AUTOTUNE = tf.data.AUTOTUNE


def _resolve_img_path(row_path: str, img_dir: str) -> str:
    """
    The CSV stores Kaggle-style absolute paths like
    `/kaggle/input/kag2/rice/rice001a.jpeg`.
    We strip everything before the class folder and rebuild against img_dir.
    """
    parts = Path(row_path).parts
    img_root = Path(img_dir)
    # Try to find the relative portion starting from a known class name
    for i, part in enumerate(parts):
        if part in CLASSES:
            rel = Path(*parts[i:])
            candidate = img_root / rel
            if candidate.exists():
                return str(candidate)
    # Fallback: use just the filename
    return str(img_root / Path(row_path).name)


def _load_image(path: str, label: int, img_size: int) -> Tuple[tf.Tensor, int]:
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def build_datasets(
    csv_path: str,
    img_dir: str,
    img_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Returns (train_ds, val_ds) from the Crop_details CSV.

    Parameters
    ----------
    csv_path : path to Crop_details.csv
    img_dir  : root directory that contains class sub-folders
               (e.g. 'data/kag2' or 'data/crop_images')
    img_size : square resize target (pixels)
    batch_size
    val_split : fraction of data to use for validation
    seed      : random seed for reproducibility
    """
    df = pd.read_csv(csv_path, index_col=0)
    df = df.dropna(subset=["path", "croplabel"])
    df["croplabel"] = df["croplabel"].astype(int)

    # Resolve image paths relative to the local img_dir
    df["local_path"] = df["path"].apply(lambda p: _resolve_img_path(p, img_dir))

    # Filter to rows whose files actually exist
    df = df[df["local_path"].apply(lambda p: Path(p).exists())]

    if len(df) == 0:
        raise FileNotFoundError(
            f"No images found in '{img_dir}'. "
            "Please copy the dataset images there."
        )

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_val = int(len(df) * val_split)

    def make_ds(subset_df: pd.DataFrame) -> tf.data.Dataset:
        paths = subset_df["local_path"].tolist()
        labels = subset_df["croplabel"].tolist()
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(
            lambda p, l: _load_image(p, l, img_size),
            num_parallel_calls=AUTOTUNE,
        )
        return ds.batch(batch_size).prefetch(AUTOTUNE)

    val_ds = make_ds(df.iloc[:n_val])
    train_ds = make_ds(df.iloc[n_val:])
    return train_ds, val_ds


def get_class_weights(csv_path: str) -> dict:
    """Compute inverse-frequency class weights for imbalanced datasets."""
    df = pd.read_csv(csv_path, index_col=0)
    counts = df["croplabel"].value_counts().sort_index()
    total = counts.sum()
    weights = {i: total / (NUM_CLASSES * cnt) for i, cnt in counts.items()}
    return weights
