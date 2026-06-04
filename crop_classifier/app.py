"""
🌾 Crop Image Classifier — Main Entry Point
Streamlit multi-page app for crop classification using CNN models.
"""

import streamlit as st

st.set_page_config(
    page_title="🌾 Crop Classifier",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ──────────────────────────────────────────────────────
st.sidebar.title("🌾 Crop Classifier")
st.sidebar.caption("AI-powered crop identification")
st.sidebar.markdown("---")

pages = {
    "🏠 Home": "pages/home.py",
    "🔍 Predict": "pages/predict.py",
    "📊 EDA": "pages/eda.py",
    "🏋️ Train": "pages/train.py",
    "📈 Metrics": "pages/metrics.py",
    "ℹ️ About": "pages/about.py",
}

# ── Home landing page ────────────────────────────────────────────────────────
st.title("🌾 Crop Image Classifier")
st.markdown(
    """
    Welcome to the **Crop Image Classifier**!  
    This tool uses a Convolutional Neural Network (CNN) trained on the  
    [CICR Crop Image Dataset](https://www.kaggle.com/) to identify **5 Indian crops**
    from field photographs.

    ---

    ### 🌿 Supported Crops
    | Label | Crop | Description |
    |-------|------|-------------|
    | 0 | **Jute** | Fibrous plant grown in warm, humid climates |
    | 1 | **Maize** | Cereal grain widely cultivated across India |
    | 2 | **Rice** | Staple food crop grown in paddy fields |
    | 3 | **Sugarcane** | Tall grass grown for sucrose production |
    | 4 | **Wheat** | Winter cereal crop grown in northern India |

    ---

    ### 🚀 Quick Start
    Use the **sidebar** to navigate between pages:

    - **Predict** — Upload an image and get an instant prediction
    - **EDA** — Explore dataset statistics and sample images
    - **Train** — Configure and train a CNN model
    - **Metrics** — View model performance (accuracy, confusion matrix)
    - **About** — Dataset & project information

    ---
    """
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Images", "1,107")
with col2:
    st.metric("Crop Classes", "5")
with col3:
    st.metric("Augmentation Types", "4 (orig, hflip, hshift, rot)")

st.info("👈 Use the sidebar links or navigate using the **Pages** menu above.")
