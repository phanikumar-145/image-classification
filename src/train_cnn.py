"""
train_cnn.py
Builds and trains a simple CNN from scratch as a baseline model.
"""

import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from src.data_preprocessing import load_datasets, prepare_datasets

MODEL_DIR = "models"
FIG_DIR = "outputs/figures"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

IMAGE_SIZE = (128, 128)
EPOCHS = 15


def build_cnn(num_classes):
    """
    A simple CNN: 3 Conv+Pool blocks, then a dense classifier head.
    Kept intentionally small — this is a baseline, not a state-of-the-art architecture.
    """
    model = models.Sequential([
        layers.Input(shape=(*IMAGE_SIZE, 3)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_cnn():
    train_ds, val_ds, test_ds, class_names = load_datasets()
    train_ds, val_ds, test_ds = prepare_datasets(train_ds, val_ds, test_ds)

    model = build_cnn(num_classes=len(class_names))
    model.summary()

    checkpoint_path = f"{MODEL_DIR}/cnn_best.keras"
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    )
    checkpoint = callbacks.ModelCheckpoint(
        checkpoint_path, monitor="val_accuracy", save_best_only=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop, checkpoint],
    )

    # Save class names so predict.py can map indices back to labels later
    with open(f"{MODEL_DIR}/class_names.json", "w") as f:
        json.dump(class_names, f)

    # Save training history for later evaluation plots
    with open(f"{MODEL_DIR}/cnn_history.json", "w") as f:
        json.dump(history.history, f)

    print(f"\nBest CNN model saved to {checkpoint_path}")
    return model, history, test_ds, class_names


if __name__ == "__main__":
    train_cnn()