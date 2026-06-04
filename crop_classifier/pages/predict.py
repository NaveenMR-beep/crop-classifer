"""
Predict page — upload a crop image and get an AI-powered classification.
"""

import io
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Predict | Crop Classifier", page_icon="🔍", layout="wide")

# ── Lazy imports so app starts even if TF is missing ─────────────────────────
@st.cache_resource
def load_model():
    """Load the trained model from disk (if available)."""
    try:
        import tensorflow as tf
        from pathlib import Path
        model_path = Path("models/crop_classifier.h5")
        if model_path.exists():
            return tf.keras.models.load_model(str(model_path))
        return None
    except Exception as e:
        st.warning(f"Model not loaded: {e}")
        return None


CLASSES = ["Jute", "Maize", "Rice", "Sugarcane", "Wheat"]
IMG_SIZE = (224, 224)

CROP_INFO = {
    "Jute": {"emoji": "🌿", "season": "Kharif", "states": "West Bengal, Bihar, Assam"},
    "Maize": {"emoji": "🌽", "season": "Kharif/Rabi", "states": "Karnataka, MP, Rajasthan"},
    "Rice": {"emoji": "🌾", "season": "Kharif", "states": "WB, UP, Punjab, AP"},
    "Sugarcane": {"emoji": "🎋", "season": "Annual", "states": "UP, Maharashtra, Tamil Nadu"},
    "Wheat": {"emoji": "🌾", "season": "Rabi", "states": "Punjab, Haryana, UP"},
}


def preprocess(image: Image.Image) -> np.ndarray:
    """Resize and normalise an image for model inference."""
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔍 Crop Predictor")
st.markdown("Upload a field photograph and the model will identify the crop.")

model = load_model()

uploaded = st.file_uploader(
    "Choose an image (JPG / JPEG / PNG)",
    type=["jpg", "jpeg", "png", "jfif"],
)

if uploaded:
    image = Image.open(uploaded)
    col_img, col_result = st.columns([1, 1])

    with col_img:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col_result:
        st.subheader("Prediction")

        if model is not None:
            with st.spinner("Running inference …"):
                x = preprocess(image)
                preds = model.predict(x)[0]
                top_idx = int(np.argmax(preds))
                top_class = CLASSES[top_idx]
                top_conf = float(preds[top_idx])

            info = CROP_INFO[top_class]
            st.success(f"**{info['emoji']} {top_class}** ({top_conf * 100:.1f}% confidence)")
            st.markdown(f"- **Season:** {info['season']}")
            st.markdown(f"- **Major States:** {info['states']}")

            st.subheader("Class Probabilities")
            for cls, prob in zip(CLASSES, preds):
                st.progress(float(prob), text=f"{cls}: {prob * 100:.1f}%")
        else:
            # Demo mode when no model is saved yet
            st.info(
                "⚠️ No trained model found at `models/crop_classifier.h5`.  \n"
                "Head to the **Train** page to train a model first."
            )
            st.markdown("**Demo prediction (random):**")
            demo_probs = np.random.dirichlet(np.ones(5))
            top_idx = int(np.argmax(demo_probs))
            top_class = CLASSES[top_idx]
            info = CROP_INFO[top_class]
            st.warning(f"🎲 Demo: {info['emoji']} **{top_class}** ({demo_probs[top_idx]*100:.1f}%)")
            for cls, prob in zip(CLASSES, demo_probs):
                st.progress(float(prob), text=f"{cls}: {prob * 100:.1f}%")
