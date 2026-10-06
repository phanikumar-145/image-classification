"""
data_preprocessing.py
Loads images from data/train, data/validation, data/test as batched,
preprocessed TensorFlow datasets, with augmentation applied only to training data.
"""

import tensorflow as tf

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

TRAIN_DIR = "data/train"
VAL_DIR = "data/validation"
TEST_DIR = "data/test"


def load_datasets():
    """Load train/validation/test datasets from directory, inferring labels from folder names."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names
    print(f"Classes found: {class_names}")

    return train_ds, val_ds, test_ds, class_names


def get_normalization_layer():
    """Rescales pixel values from [0, 255] to [0, 1]."""
    return tf.keras.layers.Rescaling(1.0 / 255)


def get_augmentation_layers():
    """
    Random augmentations applied ONLY to training data.
    Kept mild and realistic so images still represent plausible real-world variation.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ])


def prepare_datasets(train_ds, val_ds, test_ds):
    """Apply normalization to all sets, and augmentation only to training data."""
    normalization = get_normalization_layer()
    augmentation = get_augmentation_layers()

    train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y))
    train_ds = train_ds.map(lambda x, y: (normalization(x), y))

    val_ds = val_ds.map(lambda x, y: (normalization(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization(x), y))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    train_ds, val_ds, test_ds, class_names = load_datasets()
    train_ds, val_ds, test_ds = prepare_datasets(train_ds, val_ds, test_ds)

    print(f"\nNumber of classes: {len(class_names)}")
    for images, labels in train_ds.take(1):
        print(f"Batch shape: {images.shape}")
        print(f"Label shape: {labels.shape}")
        print(f"Pixel value range: {images.numpy().min():.3f} to {images.numpy().max():.3f}")