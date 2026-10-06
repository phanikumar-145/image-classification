"""
evaluate.py
Evaluates both the CNN baseline and transfer learning model on the test set.
Generates accuracy/loss curves, confusion matrices, and classification reports.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from src.data_preprocessing import load_datasets, prepare_datasets
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_DIR = "models"
FIG_DIR = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def plot_training_history(history_path, title_prefix, save_name):
    """Plot training vs validation accuracy and loss from a saved history file."""
    with open(history_path, "r") as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["accuracy"], label="Train Accuracy")
    axes[0].plot(history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title(f"{title_prefix}: Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["loss"], label="Train Loss")
    axes[1].plot(history["val_loss"], label="Validation Loss")
    axes[1].set_title(f"{title_prefix}: Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{save_name}")
    plt.close()
    print(f"Saved {save_name}")


def evaluate_model(model_path, test_ds, class_names, model_label, is_transfer_model=False):
    """Evaluate a saved model on the test set: classification report + confusion matrix."""
    model = tf.keras.models.load_model(model_path)

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    print(f"\n=== {model_label}: Classification Report ===")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix: {model_label}")
    plt.tight_layout()
    safe_name = model_label.lower().replace(" ", "_")
    plt.savefig(f"{FIG_DIR}/confusion_matrix_{safe_name}.png")
    plt.close()
    print(f"Saved confusion_matrix_{safe_name}.png")

    return report


def prepare_test_set_for_model(test_ds, is_transfer_model):
    """Apply the correct preprocessing for whichever model we're evaluating."""
    if is_transfer_model:
        return test_ds.map(lambda x, y: (preprocess_input(x), y))
    else:
        normalization = tf.keras.layers.Rescaling(1.0 / 255)
        return test_ds.map(lambda x, y: (normalization(x), y))


if __name__ == "__main__":
    _, _, test_ds_raw, class_names = load_datasets()

    # CNN baseline evaluation
    test_ds_cnn = prepare_test_set_for_model(test_ds_raw, is_transfer_model=False)
    cnn_report = evaluate_model(
        f"{MODEL_DIR}/cnn_best.keras", test_ds_cnn, class_names, "CNN Baseline"
    )
    plot_training_history(f"{MODEL_DIR}/cnn_history.json", "CNN Baseline", "cnn_training_curves.png")

    # Transfer learning evaluation (need a fresh copy of the raw test set — datasets are single-use iterators)
    _, _, test_ds_raw2, _ = load_datasets()
    test_ds_transfer = prepare_test_set_for_model(test_ds_raw2, is_transfer_model=True)
    transfer_report = evaluate_model(
        f"{MODEL_DIR}/transfer_best.keras", test_ds_transfer, class_names, "Transfer Learning"
    )
    plot_training_history(f"{MODEL_DIR}/transfer_history.json", "Transfer Learning", "transfer_training_curves.png")

    # Save both reports for Step 11's comparison table
    with open(f"{MODEL_DIR}/evaluation_reports.json", "w") as f:
        json.dump({"cnn": cnn_report, "transfer": transfer_report}, f, indent=2)

    print("\nEvaluation complete. Reports saved to models/evaluation_reports.json")