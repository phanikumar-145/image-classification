"""
train_transfer_learning.py
Builds and trains a transfer learning model using MobileNetV2 pretrained on ImageNet.
"""

import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.data_preprocessing import load_datasets

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

IMAGE_SIZE = (128, 128)
EPOCHS_HEAD = 10
EPOCHS_FINE_TUNE = 5


def prepare_for_transfer_learning(train_ds, val_ds, test_ds):
    """
    MobileNetV2 expects its own specific preprocessing (pixel values scaled to [-1, 1]),
    NOT the [0, 1] rescaling we used for the CNN baseline. We apply augmentation to
    training data only, then MobileNetV2's own preprocess_input to all sets.
    """
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ])

    train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y))
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))
    test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


def build_transfer_model(num_classes):
    """
    MobileNetV2 pretrained on ImageNet as a frozen feature extractor,
    with a new classification head trained from scratch on top.
    """
    base_model = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,       # exclude ImageNet's original 1000-class output layer
        weights="imagenet",
    )
    base_model.trainable = False  # freeze the pretrained base initially

    model = models.Sequential([
        base_model,
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
    return model, base_model


def train_transfer_learning():
    train_ds, val_ds, test_ds, class_names = load_datasets()
    train_ds, val_ds, test_ds = prepare_for_transfer_learning(train_ds, val_ds, test_ds)

    model, base_model = build_transfer_model(num_classes=len(class_names))
    model.summary()

    checkpoint_path = f"{MODEL_DIR}/transfer_best.keras"
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    checkpoint = callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True)

    print("\n--- Phase 1: Training classification head (base frozen) ---")
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=[early_stop, checkpoint],
    )

    print("\n--- Phase 2: Fine-tuning (unfreezing top layers of base model) ---")
    base_model.trainable = True
    # Only unfreeze the last ~20 layers, keep early layers (generic features) frozen
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Recompile with a much lower learning rate — critical for fine-tuning,
    # to avoid destroying the pretrained weights with large updates
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE_TUNE,
        callbacks=[early_stop, checkpoint],
    )

    combined_history = {
        key: history_head.history[key] + history_fine.history[key]
        for key in history_head.history
    }
    with open(f"{MODEL_DIR}/transfer_history.json", "w") as f:
        json.dump(combined_history, f)

    print(f"\nBest transfer learning model saved to {checkpoint_path}")
    return model, combined_history, test_ds, class_names


if __name__ == "__main__":
    train_transfer_learning()