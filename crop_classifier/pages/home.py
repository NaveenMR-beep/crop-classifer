"""Home page — shown when the user navigates to /home."""

import streamlit as st

st.set_page_config(page_title="Home | Crop Classifier", page_icon="🏠", layout="wide")

st.title("🏠 Home")
st.markdown(
    """
    ### About this Project

    This application classifies crop field images into one of **five categories** using a
    fine-tuned **MobileNetV2** backbone (transfer learning).

    #### Dataset structure
    ```
    archive/
    ├── Crop_details.csv          ← labels (path, crop, croplabel)
    ├── crop_images/              ← 40 original images per class
    │   ├── jute/
    │   ├── maize/
    │   ├── rice/
    │   ├── sugarcane/
    │   └── wheat/
    ├── kag2/                     ← augmented images (×4 per original)
    │   └── ... (hflip, hshift, rot variants)
    ├── some_more_images/         ← 10 extra images per class (41–50)
    └── test_crop_image/          ← unlabelled test images
    ```

    #### Pipeline
    ```
    Raw images → Preprocessing → Augmentation → CNN Training → Evaluation → Deployment
    ```
    """
)
