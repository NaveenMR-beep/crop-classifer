"""
EDA page — explore the dataset: class distribution, sample images, augmentations.
"""

import os
import random
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="EDA | Crop Classifier", page_icon="📊", layout="wide")

st.title("📊 Exploratory Data Analysis")

# ── Locate data ───────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "Crop_details.csv"

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


# ── Dataset overview ──────────────────────────────────────────────────────────
st.subheader("Dataset Overview")

if CSV_PATH.exists():
    df = load_csv(CSV_PATH)
    st.dataframe(df.head(20), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Distribution")
        counts = df["crop"].value_counts()
        st.bar_chart(counts)

    with col2:
        st.subheader("Label Map")
        label_map = df[["crop", "croplabel"]].drop_duplicates().sort_values("croplabel")
        st.dataframe(label_map, use_container_width=True)

    st.subheader("Summary Statistics")
    st.write(f"- **Total samples:** {len(df):,}")
    st.write(f"- **Classes:** {df['crop'].nunique()}")
    st.write(f"- **Samples per class:** {dict(counts)}")
else:
    st.warning(
        "CSV not found at `data/Crop_details.csv`.  \n"
        "Please place `Crop_details.csv` inside the `data/` directory."
    )

# ── Sample images ─────────────────────────────────────────────────────────────
st.subheader("Sample Images from Each Class")
CROPS = ["jute", "maize", "rice", "sugarcane", "wheat"]

image_dir = DATA_DIR / "crop_images"
if image_dir.exists():
    cols = st.columns(len(CROPS))
    for col, crop in zip(cols, CROPS):
        with col:
            crop_dir = image_dir / crop
            imgs = list(crop_dir.glob("*.jpeg")) + list(crop_dir.glob("*.jpg"))
            if imgs:
                img_path = random.choice(imgs)
                st.image(str(img_path), caption=crop.title(), use_container_width=True)
            else:
                st.caption(f"No images found for {crop}")
else:
    st.info("Place the extracted dataset images in `data/crop_images/<class>/` to view samples.")

# ── Augmentation demo ─────────────────────────────────────────────────────────
st.subheader("Augmentation Variants (kag2 folder)")
st.markdown(
    """
    Each original image in the `kag2/` directory has **4 variants**:
    | Suffix | Transform |
    |--------|-----------|
    | `a`    | Original  |
    | `ahf`  | Horizontal flip |
    | `ahs`  | Horizontal shift |
    | `arot` | Rotation  |
    """
)

kag2_dir = DATA_DIR / "kag2"
if kag2_dir.exists():
    selected_crop = st.selectbox("Select a crop to preview augmentations", CROPS)
    crop_dir = kag2_dir / selected_crop
    if crop_dir.exists():
        # Find a base image (original 'a' variant)
        originals = sorted(crop_dir.glob("*[0-9]a.jpeg"))
        if originals:
            base = random.choice(originals[:10])
            stem = base.stem.replace("a", "")
            variants = {
                "Original": base,
                "H-Flip": crop_dir / f"{stem}ahf.jpeg",
                "H-Shift": crop_dir / f"{stem}ahs.jpeg",
                "Rotation": crop_dir / f"{stem}arot.jpeg",
            }
            cols = st.columns(4)
            for col, (label, path) in zip(cols, variants.items()):
                with col:
                    if path.exists():
                        st.image(str(path), caption=label, use_container_width=True)
                    else:
                        st.caption(f"{label} not found")
else:
    st.info("Place the `kag2/` folder inside `data/` to preview augmentations.")
