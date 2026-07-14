# 🩺 Pancreatic Cancer Detection using Deep Learning

A modular deep learning framework for **automatic pancreatic tumor detection** from CT scan images using state-of-the-art transfer learning models including **DenseNet121**, **EfficientNetB0**, and **Xception**. The project supports transfer learning, fine-tuning, K-Fold Cross Validation, model evaluation, visualization, and single-image inference.

---

# 📌 Table of Contents

- Project Overview
- Features
- Project Structure
- Dataset
- Technologies Used
- Model Architecture
- Training Pipeline
- Transfer Learning Strategy
- Evaluation Metrics
- Results
- Installation
- Usage
- Prediction
- Configuration
- Future Improvements
- License
- Author

---

# 📖 Project Overview

Pancreatic cancer is among the deadliest cancers because it is often diagnosed at an advanced stage. Early detection using medical imaging and deep learning can significantly assist clinicians in diagnosis.

This project provides a complete deep learning pipeline capable of:

- Loading CT scan datasets
- Data preprocessing and augmentation
- Transfer learning using pretrained CNNs
- Fine-tuning pretrained models
- Model evaluation
- Confusion matrix visualization
- Classification reports
- Single-image prediction
- Modular codebase suitable for research and deployment

---

# ✨ Features

✅ Modular project architecture

✅ DenseNet121

✅ EfficientNetB0

✅ Xception

✅ Transfer Learning

✅ Fine Tuning

✅ Early Stopping

✅ Learning Rate Scheduling

✅ Data Augmentation

✅ K-Fold Cross Validation

✅ Confusion Matrix

✅ Classification Report

✅ Single Image Prediction

✅ Model Saving & Loading

---

# 📂 Project Structure

```
PancreaticCancerDetection
│
├── config.py
├── main.py
├── requirements.txt
├── README.md
│
├── data
│   ├── dataset.py
│   └── dataloader.py
│
├── models
│   ├── densenet.py
│   ├── efficientnet.py
│   ├── xception.py
│   └── trainer.py
│
├── evaluation
│   ├── evaluator.py
│   └── metrics.py
│
├── prediction
│   └── predictor.py
│
├── visualization
│   ├── plots.py
│   └── confusion_matrix.py
│
├── utils
│   ├── callbacks.py
│   ├── helper.py
│   ├── labels.py
│   └── seed.py
│
├── saved_models
│
└── dataset
    ├── train
    └── test
```

---

# 📊 Dataset

The project expects the following dataset organization.

```
dataset

├── train
│     ├── normal
│     └── pancreatic_tumor
│
└── test
      ├── normal
      └── pancreatic_tumor
```

The dataset used for this work is **not included** in this repository due to privacy and licensing restrictions.

---

# 🛠 Technologies Used

- Python 3.11
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- Pillow

---

# 🧠 Supported Deep Learning Models

## DenseNet121

- ImageNet pretrained weights
- Global Average Pooling
- Dropout
- Dense classifier

## EfficientNetB0

- Efficient scaling
- Transfer learning
- Fine tuning

## Xception

- Depthwise separable convolutions
- ImageNet pretrained weights
- High accuracy feature extraction

---

# 🔄 Training Pipeline

```
CT Scan Images
        │
        ▼
Data Loading
        │
        ▼
Image Augmentation
        │
        ▼
Transfer Learning
        │
        ▼
Model Training
        │
        ▼
Fine Tuning
        │
        ▼
Evaluation
        │
        ▼
Prediction
```

---

# ⚙️ Transfer Learning Strategy

### Phase 1

- Freeze pretrained backbone
- Train classifier head

### Phase 2

- Unfreeze top layers
- Fine tune entire network using a smaller learning rate

---

# 📈 Evaluation Metrics

The following evaluation metrics are calculated:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report
- Sensitivity
- Specificity

---

# 📉 Visualizations

The project automatically generates:

- Training Accuracy Curve
- Validation Accuracy Curve
- Training Loss Curve
- Validation Loss Curve
- Confusion Matrix
- Prediction Probability Graph

---

# 💾 Model Saving

The trained model is automatically saved as

```
saved_models/
    pancreatic_model.keras
```

Class labels are also stored for future inference.

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/PancreaticCancerDetection.git
```

Move inside project

```bash
cd PancreaticCancerDetection
```

Create virtual environment

```bash
python -m venv .venv
```

Activate

Windows

```bash
.venv\Scripts\activate
```

Linux

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Simply execute

```bash
python main.py
```

The pipeline automatically performs

- Dataset loading
- Data augmentation
- Model creation
- Transfer learning
- Fine tuning
- Evaluation
- Visualization
- Model saving

---

# 🔍 Predict on a New Image

```python
from prediction.predictor import Predictor

predictor = Predictor()

label = predictor.predict("sample.jpg")

print(label)
```

---

# ⚙ Configuration

All configurable parameters are available inside

```
config.py
```

Examples include

- Batch size
- Learning rate
- Epochs
- Image size
- Dataset paths
- Model selection

---

# 📈 Example Results

Example evaluation outputs include:

- Training Accuracy
- Validation Accuracy
- Test Accuracy
- Precision
- Recall
- Confusion Matrix
- Classification Report

*(Replace this section with your actual experimental results once finalized.)*

---

# 🔮 Future Improvements

- Vision Transformers (ViT)
- Grad-CAM visualization
- Attention mechanisms
- Hyperparameter optimization
- Multi-class pancreatic disease classification
- Web deployment using Flask/FastAPI
- Docker containerization
- ONNX model export

---

# 🤝 Contributing

Contributions are welcome.

Please fork the repository, create a feature branch, and submit a pull request.

---

# 📜 License

This project is intended for **educational and research purposes**.

The dataset is not distributed through this repository.

---

# 👩‍💻 Author

**Himanshi**

- B.Tech in Communication & Computer Engineering (Honors in AI & DS)
- Deep Learning • Medical Imaging • Computer Vision

---

# ⭐ Acknowledgements

- TensorFlow
- Keras
- Scikit-Learn
- ImageNet Pretrained Models