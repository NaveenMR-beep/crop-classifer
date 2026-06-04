"""
Train page — configure hyperparameters and train a MobileNetV2 crop classifier.
"""

import streamlit as st

st.set_page_config(page_title="Train | Crop Classifier", page_icon="🏋️", layout="wide")

st.title("🏋️ Model Training")
st.markdown("Configure the training parameters and train a CNN on the crop dataset.")

# ── Configuration sidebar ─────────────────────────────────────────────────────
with st.expander("⚙️ Training Configuration", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.slider("Epochs", 1, 50, 10)
        batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=2)
        img_size = st.selectbox("Image Size", [128, 160, 224], index=2)
    with col2:
        lr = st.select_slider(
            "Learning Rate",
            options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            value=1e-4,
            format_func=lambda x: f"{x:.0e}",
        )
        val_split = st.slider("Validation Split", 0.1, 0.4, 0.2, 0.05)
        use_augmentation = st.checkbox("Use pre-augmented (kag2) images", value=True)
        fine_tune = st.checkbox("Fine-tune backbone (last 30 layers)", value=False)

st.markdown("---")

# ── Architecture info ─────────────────────────────────────────────────────────
st.subheader("📐 Architecture: MobileNetV2 + Custom Head")
st.markdown(
    """
    ```
    Input (224×224×3)
        ↓
    MobileNetV2 backbone (ImageNet weights, frozen)
        ↓
    GlobalAveragePooling2D
        ↓
    Dense(256, relu)  →  Dropout(0.4)
        ↓
    Dense(5, softmax)   ← 5 crop classes
    ```
    """
)

# ── Training code preview ─────────────────────────────────────────────────────
with st.expander("📄 View training code (`models/train.py`)"):
    st.code(
        """
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from utils.data_loader import build_datasets

def build_model(num_classes=5, img_size=224, fine_tune=False):
    base = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = fine_tune
    if fine_tune:
        for layer in base.layers[:-30]:
            layer.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(base.input, outputs)

def train(config):
    train_ds, val_ds = build_datasets(
        csv_path=config["csv_path"],
        img_dir=config["img_dir"],
        img_size=config["img_size"],
        batch_size=config["batch_size"],
        val_split=config["val_split"],
    )
    model = build_model(fine_tune=config["fine_tune"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config["lr"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "models/crop_classifier.h5", save_best_only=True, monitor="val_accuracy"
        ),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
    ]
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=config["epochs"], callbacks=callbacks
    )
    return model, history
""",
        language="python",
    )

# ── Launch training button ────────────────────────────────────────────────────
st.markdown("---")
if st.button("🚀 Start Training", type="primary"):
    try:
        import tensorflow as tf
        from pathlib import Path
        from utils.data_loader import build_datasets
        from models.train import build_model

        DATA_DIR = Path("data")
        csv_path = DATA_DIR / "Crop_details.csv"
        img_dir = DATA_DIR / ("kag2" if use_augmentation else "crop_images")

        if not csv_path.exists():
            st.error("CSV not found at `data/Crop_details.csv`. Please add the dataset first.")
            st.stop()

        config = dict(
            csv_path=str(csv_path),
            img_dir=str(img_dir),
            img_size=img_size,
            batch_size=batch_size,
            val_split=val_split,
            lr=lr,
            epochs=epochs,
            fine_tune=fine_tune,
        )

        st.info("Building datasets …")
        train_ds, val_ds = build_datasets(**{k: v for k, v in config.items()
                                              if k in ["csv_path", "img_dir", "img_size",
                                                       "batch_size", "val_split"]})

        model = build_model(fine_tune=fine_tune, img_size=img_size)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        progress = st.progress(0, "Training …")
        log_area = st.empty()

        history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

        for epoch in range(epochs):
            result = model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=0)
            for k in history:
                history[k].append(result.history[k][0])
            progress.progress((epoch + 1) / epochs, f"Epoch {epoch+1}/{epochs}")
            log_area.dataframe(
                {k: [round(v[-1], 4)] for k, v in history.items()},
                use_container_width=True,
            )

        Path("models").mkdir(exist_ok=True)
        model.save("models/crop_classifier.h5")
        st.success("✅ Training complete! Model saved to `models/crop_classifier.h5`.")
        st.session_state["history"] = history

    except ImportError:
        st.error("TensorFlow is required for training. Install it with `pip install tensorflow`.")
    except Exception as e:
        st.error(f"Training failed: {e}")
