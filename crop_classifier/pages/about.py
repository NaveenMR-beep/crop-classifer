"""About page — dataset credits and tech stack."""

import streamlit as st

st.set_page_config(page_title="About | Crop Classifier", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About")

st.markdown(
    """
    ## 🌾 Crop Image Classifier

    This project is an end-to-end **image classification pipeline** for identifying
    five major Indian crops from field photographs.

    ---

    ### 📦 Dataset
    - **Source:** Kaggle — *Crop Image Classification Dataset (CICR)*
    - **Classes:** Jute, Maize, Rice, Sugarcane, Wheat
    - **Images:** ~200 per class (40 originals × 4 augmentations via `kag2/`) + 10 extras
    - **CSV:** `Crop_details.csv` maps image paths to crop labels

    ---

    ### 🛠️ Tech Stack
    | Layer | Technology |
    |-------|-----------|
    | UI | Streamlit |
    | Model | TensorFlow / Keras (MobileNetV2) |
    | Data | Pandas, NumPy |
    | Metrics | scikit-learn |
    | Packaging | Python 3.10+ |

    ---

    ### 📁 Project Structure
    ```
    crop_classifier/
    ├── app.py                  ← Streamlit entry point
    ├── pages/
    │   ├── home.py
    │   ├── predict.py
    │   ├── eda.py
    │   ├── train.py
    │   ├── metrics.py
    │   └── about.py
    ├── utils/
    │   ├── data_loader.py      ← Dataset pipeline
    │   └── image_utils.py      ← Preprocessing helpers
    ├── models/
    │   ├── train.py            ← Model definition & training loop
    │   └── crop_classifier.h5  ← Saved model (generated after training)
    ├── data/
    │   ├── Crop_details.csv    ← Labels (place dataset here)
    │   ├── crop_images/        ← Original images
    │   └── kag2/               ← Augmented images
    ├── notebooks/
    │   └── exploration.ipynb   ← EDA notebook
    ├── assets/
    │   └── banner.png          ← UI assets
    ├── requirements.txt
    ├── .gitignore
    └── README.md
    ```

    ---

    ### 🚀 Running Locally
    ```bash
    git clone https://github.com/<your-username>/crop-classifier.git
    cd crop-classifier
    pip install -r requirements.txt

    # Place the dataset inside data/
    unzip archive.zip -d data/

    streamlit run app.py
    ```

    ---

    ### 📜 License
    MIT License — free for educational and research use.
    """
)
