"""
Metrics page — training history curves and confusion matrix.
"""

import numpy as np
import streamlit as st

st.set_page_config(page_title="Metrics | Crop Classifier", page_icon="📈", layout="wide")

st.title("📈 Model Metrics")

CLASSES = ["Jute", "Maize", "Rice", "Sugarcane", "Wheat"]


def plot_history(history: dict):
    """Render training/validation accuracy and loss curves."""
    import pandas as pd

    epochs = range(1, len(history["accuracy"]) + 1)
    acc_df = pd.DataFrame(
        {"Train Accuracy": history["accuracy"], "Val Accuracy": history["val_accuracy"]},
        index=list(epochs),
    )
    loss_df = pd.DataFrame(
        {"Train Loss": history["loss"], "Val Loss": history["val_loss"]},
        index=list(epochs),
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Accuracy")
        st.line_chart(acc_df)
    with col2:
        st.subheader("Loss")
        st.line_chart(loss_df)


def plot_confusion_matrix(cm: np.ndarray):
    """Display a confusion matrix as a styled dataframe."""
    import pandas as pd

    df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    st.subheader("Confusion Matrix")
    st.dataframe(df.style.background_gradient(cmap="Blues"), use_container_width=True)


# ── Show training history if available ───────────────────────────────────────
if "history" in st.session_state:
    st.subheader("Training History")
    plot_history(st.session_state["history"])
else:
    st.info("No training history found. Train a model on the **Train** page first.")

st.markdown("---")

# ── Evaluate on test set ──────────────────────────────────────────────────────
st.subheader("Evaluate Model on Test Set")

if st.button("Run Evaluation"):
    try:
        import tensorflow as tf
        from pathlib import Path
        from utils.data_loader import build_datasets

        model_path = Path("models/crop_classifier.h5")
        csv_path = Path("data/Crop_details.csv")

        if not model_path.exists():
            st.error("No trained model found. Train one first.")
        elif not csv_path.exists():
            st.error("CSV not found at `data/Crop_details.csv`.")
        else:
            model = tf.keras.models.load_model(str(model_path))
            _, val_ds = build_datasets(
                csv_path=str(csv_path),
                img_dir="data/kag2",
                img_size=224,
                batch_size=32,
                val_split=0.2,
            )

            y_true, y_pred = [], []
            for imgs, labels in val_ds:
                preds = model.predict(imgs, verbose=0)
                y_pred.extend(np.argmax(preds, axis=1))
                y_true.extend(labels.numpy())

            from sklearn.metrics import (
                classification_report,
                confusion_matrix,
                accuracy_score,
            )

            acc = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            report = classification_report(y_true, y_pred, target_names=CLASSES)

            st.metric("Validation Accuracy", f"{acc * 100:.2f}%")
            plot_confusion_matrix(cm)

            st.subheader("Classification Report")
            st.text(report)

    except ImportError as e:
        st.error(f"Missing dependency: {e}")
    except Exception as e:
        st.error(f"Evaluation failed: {e}")

# ── Demo confusion matrix ─────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🎲 Show Demo Confusion Matrix"):
    np.random.seed(42)
    demo_cm = np.random.randint(0, 50, (5, 5))
    np.fill_diagonal(demo_cm, np.random.randint(150, 200, 5))
    plot_confusion_matrix(demo_cm)
