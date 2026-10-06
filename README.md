# Image Classification using CNN and Transfer Learning

## Project Overview

This project builds and compares two image classification approaches: a CNN trained
from scratch, and a transfer learning model using MobileNetV2 pretrained on ImageNet.
Both classify images into 5 clothing categories.

## Problem Statement

Build a system that can classify an image into one of several predefined categories,
and demonstrate whether transfer learning provides a meaningful advantage over a
custom CNN trained from scratch on a small dataset.

## Objective

Classify images into 5 clothing categories using a CNN baseline and a transfer
learning model, compare their performance with real evaluation metrics, and understand
the trade-offs between the two approaches.

## Dataset

**Fashion-MNIST** (via `tensorflow.keras.datasets.fashion_mnist`), using 5 of its
10 classes:

| Class | Train images | Validation images | Test images |
|---|---|---|---|
| bag | 500 | 100 | 100 |
| pullover | 500 | 100 | 100 |
| sneaker | 500 | 100 | 100 |
| trouser | 500 | 100 | 100 |
| tshirt | 500 | 100 | 100 |

Images were resized from their native 28×28 to 128×128 for training. The dataset is
intentionally small and balanced (equal images per class) to keep CPU training fast
while still being large enough to demonstrate meaningful learning.

**Note:** the dataset is not committed to this repository (see `.gitignore`). Run
`python -m src.prepare_dataset` to regenerate it automatically — no manual download
required.

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2 (transfer learning, ImageNet pretrained weights)
- NumPy, Matplotlib
- Scikit-learn (evaluation metrics)
- Pillow, OpenCV
- Streamlit

## Project Structure

image-classification/
│
├── data/ (generated, not committed)
│ ├── train/
│ ├── validation/
│ └── test/
│
├── models/
│ ├── cnn_best.keras
│ ├── transfer_best.keras
│ ├── class_names.json
│ ├── cnn_history.json
│ ├── transfer_history.json
│ └── evaluation_reports.json
│
├── src/
│ ├── prepare_dataset.py
│ ├── data_preprocessing.py
│ ├── explore_data.py
│ ├── train_cnn.py
│ ├── train_transfer_learning.py
│ ├── evaluate.py
│ ├── compare_models.py
│ └── predict.py
│
├── outputs/
│ └── figures/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore


## Data Preprocessing

- Images resized to 128×128
- Pixel values normalized: `[0, 1]` scaling for the CNN baseline, MobileNetV2's own
  `[-1, 1]` `preprocess_input` scaling for the transfer learning model
- Data augmentation (horizontal flip, small rotation, small zoom) applied only to
  training data — never to validation or test data, to keep evaluation realistic
- Dataset built as balanced (equal images per class) to avoid bias toward any one class

## CNN Architecture (Baseline)

3 convolutional blocks (Conv2D + MaxPooling2D, 32→64→128 filters), followed by
GlobalAveragePooling2D, a Dense(128) layer, Dropout(0.3), and a softmax output layer.
Trained from scratch with Adam optimizer, categorical crossentropy loss, early
stopping, and model checkpointing.

## Transfer Learning

MobileNetV2, pretrained on ImageNet, used as a frozen feature extractor with a new
classification head trained on top. Trained in two phases:
1. **Head training** — base model frozen, only the new classification head trained
2. **Fine-tuning** — last 20 layers of MobileNetV2 unfrozen and trained with a much
   lower learning rate (1e-5) to gently adapt high-level features without destroying
   pretrained knowledge

## Data Augmentation

Random horizontal flip, small rotation (±5%), and small zoom (±10%) — kept
deliberately mild so augmented images still represent plausible real-world variation
of the same clothing item.

## Model Training

| Model | Epochs | Final Train Accuracy | Final Val Accuracy |
|---|---|---|---|
| CNN Baseline | 15 | 0.896 | 0.930 |
| Transfer Learning (head + fine-tune) | 10 + 5 | 0.973 | 0.960 |

## Evaluation

Evaluated on a held-out test set (500 images, 100 per class) using precision, recall,
F1-score (per class and macro-averaged), and confusion matrices.

## Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| CNN Baseline | 0.932 | 0.934 | 0.932 | 0.931 |
| **Transfer Learning** | **0.966** | **0.967** | **0.966** | **0.966** |

**Per-class F1-score:**

| Class | CNN | Transfer Learning |
|---|---|---|
| bag | 0.929 | 0.970 |
| pullover | 0.915 | 0.933 |
| sneaker | 0.975 | 0.990 |
| trouser | 0.966 | 0.995 |
| tshirt | 0.871 | 0.944 |

Transfer learning outperformed the CNN baseline on every single class, with the
largest improvement on `tshirt` — the hardest class for both models, most likely due
to visual similarity with `pullover`.

## Key Insights

- Transfer learning consistently outperformed the CNN baseline, but not by an
  overwhelming margin — the CNN baseline already performed well (93%) on this
  relatively simple, visually distinct dataset
- Both models struggled most with distinguishing `tshirt` from `pullover` — a
  sensible confusion, since these two clothing types share a similar silhouette
- Neither model showed strong evidence of overfitting; train/validation accuracy
  gaps stayed small throughout training
- **The model does not generalize well to real-world photos** outside its training
  distribution (tested manually — see Future Improvements)

## Prediction

```powershell
python -m src.predict "path/to/image.jpg"
```

Outputs the predicted class, confidence percentage, and the full probability
distribution across all 5 classes.

**Important limitation, confirmed through testing:** the model performs very well on
images similar to its training data (simple, centered, plain-background clothing
item photos), but produces confidently wrong predictions on real-world photos with
different backgrounds, lighting, and textures — a known failure mode of models
trained on narrow, simple datasets, not a bug in the code.

## Streamlit Application

```powershell
streamlit run app.py
```

Upload an image, view it, and get a predicted class with a confidence score and
full per-class probability breakdown.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run

```powershell
# 1. Generate the dataset (one-time, auto-downloads Fashion-MNIST)
python -m src.prepare_dataset

# 2. Explore the data (optional)
python -m src.explore_data

# 3. Train the CNN baseline
python -m src.train_cnn

# 4. Train the transfer learning model
python -m src.train_transfer_learning

# 5. Evaluate both models
python -m src.evaluate

# 6. Compare results
python -m src.compare_models

# 7. Predict on a new image
python -m src.predict "path/to/image.jpg"

# 8. Run the interactive app
streamlit run app.py
```

## Future Improvements

- Train on a more diverse, real-world image dataset (varied backgrounds, lighting,
  angles) to close the generalization gap observed with real photos
- Try a heavier pretrained model (e.g., EfficientNetB0) for comparison
- Add a "confidence threshold" in the app to flag low-confidence or
  out-of-distribution predictions rather than always forcing a top-1 answer
- Address the tshirt/pullover confusion specifically, e.g., with additional
  training images emphasizing their distinguishing features