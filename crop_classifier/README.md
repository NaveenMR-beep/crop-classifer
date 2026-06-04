# 🌾 Crop Image Classifier

A Streamlit web application that classifies field photographs into **5 Indian crop types**
using transfer learning (MobileNetV2).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📸 Screenshots

| Predict | EDA | Metrics |
|---------|-----|---------|
| Upload → instant class probabilities | Dataset stats & sample gallery | Accuracy curves & confusion matrix |

---

## 🌿 Supported Crops

| Label | Crop | Season |
|-------|------|--------|
| 0 | Jute | Kharif |
| 1 | Maize | Kharif / Rabi |
| 2 | Rice | Kharif |
| 3 | Sugarcane | Annual |
| 4 | Wheat | Rabi |

---

## 📦 Dataset

The model is trained on the **CICR Crop Image Dataset** (Kaggle).

```
data/
├── Crop_details.csv          ← label file (committed to repo)
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

> **Large files** (`crop_images/`, `kag2/`, `some_more_images/`) are listed in `.gitignore`.
> Download them from Kaggle and place them in `data/`.

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/crop-classifier.git
cd crop-classifier
```

### 2. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add the dataset
```bash
# Download from Kaggle and unzip into data/
unzip /path/to/archive.zip -d data/
```

### 4. Launch the app
```bash
streamlit run app.py
```

---

## 🏋️ Training from the CLI

```bash
python -m models.train \
  --csv data/Crop_details.csv \
  --img_dir data/kag2 \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4 \
  --fine_tune
```

The best checkpoint is saved to `models/crop_classifier.h5`.

---

## 📁 Project Structure

```
crop_classifier/
├── app.py                     ← Streamlit entry point
├── pages/
│   ├── home.py                ← Landing / overview
│   ├── predict.py             ← Upload & classify
│   ├── eda.py                 ← Dataset exploration
│   ├── train.py               ← Interactive training UI
│   ├── metrics.py             ← Accuracy, confusion matrix
│   └── about.py               ← Credits & tech stack
├── utils/
│   ├── __init__.py
│   ├── data_loader.py         ← tf.data pipeline
│   └── image_utils.py         ← Preprocessing helpers
├── models/
│   ├── __init__.py
│   ├── train.py               ← Model definition + training loop
│   └── crop_classifier.h5     ← Trained weights (not committed)
├── data/
│   └── Crop_details.csv       ← Label CSV (committed)
├── notebooks/
│   └── exploration.ipynb      ← EDA notebook
├── assets/
├── .streamlit/
│   └── config.toml            ← Green theme
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧠 Model Architecture

```
Input (224×224×3)
    ↓
MobileNetV2 (ImageNet pretrained, frozen backbone)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, relu) → Dropout(0.4)
    ↓
Dense(5, softmax)
```

**Fine-tuning** (optional): the last 30 MobileNetV2 layers can be unfrozen
for a second training phase at a lower learning rate.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Val Accuracy | ~92% |
| Val Loss | ~0.22 |
| Params (head only) | ~264K |
| Total params | ~3.5M |

> Results vary by hardware, random seed, and whether augmented data (`kag2/`) is used.

---

## 📜 License

MIT © 2024 — free to use for educational and research purposes.

---

## 🙏 Acknowledgements

- Dataset: [Crop Image Classification Dataset — Kaggle](https://www.kaggle.com/)
- Backbone: [MobileNetV2 — Google](https://arxiv.org/abs/1801.04381)
- UI: [Streamlit](https://streamlit.io)
