"""
models/train.py
───────────────
MobileNetV2-based crop classifier — model definition and training helpers.

Usage (standalone):
    python -m models.train --csv data/Crop_details.csv --img_dir data/kag2 --epochs 20
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

NUM_CLASSES = 5
DEFAULT_IMG_SIZE = 224


def build_model(
    num_classes: int = NUM_CLASSES,
    img_size: int = DEFAULT_IMG_SIZE,
    fine_tune: bool = False,
    dropout_rate: float = 0.4,
) -> Model:
    """
    Build a transfer-learning model using MobileNetV2 as the backbone.

    Parameters
    ----------
    num_classes   : number of output classes
    img_size      : square input size (pixels)
    fine_tune     : if True, unfreeze the last 30 backbone layers
    dropout_rate  : dropout probability in the classification head
    """
    base = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # freeze backbone initially

    if fine_tune:
        # Unfreeze the top N layers for fine-tuning
        for layer in base.layers[:-30]:
            layer.trainable = False
        for layer in base.layers[-30:]:
            layer.trainable = True

    x = base.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return Model(inputs=base.input, outputs=outputs, name="CropClassifier")


def compile_model(
    model: Model,
    learning_rate: float = 1e-4,
) -> Model:
    """Compile with Adam + sparse cross-entropy."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks(model_dir: str = "models") -> list:
    """Standard training callbacks."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{model_dir}/crop_classifier.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f"{model_dir}/logs",
            histogram_freq=0,
        ),
    ]


def train(
    csv_path: str,
    img_dir: str,
    img_size: int = DEFAULT_IMG_SIZE,
    batch_size: int = 32,
    epochs: int = 20,
    val_split: float = 0.2,
    learning_rate: float = 1e-4,
    fine_tune: bool = False,
    model_dir: str = "models",
) -> Tuple[Model, dict]:
    """Full training pipeline. Returns (model, history.history)."""
    from utils.data_loader import build_datasets, get_class_weights

    print(f"[train] Loading data from {img_dir} …")
    train_ds, val_ds = build_datasets(
        csv_path=csv_path,
        img_dir=img_dir,
        img_size=img_size,
        batch_size=batch_size,
        val_split=val_split,
    )

    class_weights = get_class_weights(csv_path)
    print(f"[train] Class weights: {class_weights}")

    model = build_model(img_size=img_size, fine_tune=fine_tune)
    model = compile_model(model, learning_rate=learning_rate)
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=get_callbacks(model_dir),
        verbose=1,
    )

    return model, history.history


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Crop Classifier")
    parser.add_argument("--csv", default="data/Crop_details.csv")
    parser.add_argument("--img_dir", default="data/kag2")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--model_dir", default="models")
    args = parser.parse_args()

    train(
        csv_path=args.csv,
        img_dir=args.img_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_split=args.val_split,
        learning_rate=args.lr,
        fine_tune=args.fine_tune,
        model_dir=args.model_dir,
    )
