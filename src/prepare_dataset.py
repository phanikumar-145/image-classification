"""
prepare_dataset.py
Downloads a subset of Fashion-MNIST (via Keras) and exports it into
data/train/<class>/, data/validation/<class>/, data/test/<class>/ folders.
"""

import os
import numpy as np
from PIL import Image
from tensorflow.keras.datasets import fashion_mnist

# Fashion-MNIST's full label list, for reference:
# 0=T-shirt/top, 1=Trouser, 2=Pullover, 3=Dress, 4=Coat,
# 5=Sandal, 6=Shirt, 7=Sneaker, 8=Bag, 9=Ankle boot
SELECTED_CLASSES = {0: "tshirt", 1: "trouser", 2: "pullover", 7: "sneaker", 8: "bag"}

IMAGES_PER_CLASS_TRAIN = 500
IMAGES_PER_CLASS_VAL = 100
IMAGES_PER_CLASS_TEST = 100

OUTPUT_DIR = "data"


def save_images(images, labels, split_name, per_class_limit, start_offset=0):
    for class_id, class_name in SELECTED_CLASSES.items():
        class_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
        os.makedirs(class_dir, exist_ok=True)

        class_indices = np.where(labels.flatten() == class_id)[0]
        selected = class_indices[start_offset:start_offset + per_class_limit]

        for i, idx in enumerate(selected):
            img = Image.fromarray(images[idx])
            img.save(os.path.join(class_dir, f"{class_name}_{i}.jpg"))

        print(f"Saved {len(selected)} images to {class_dir}")


def prepare_dataset():
    print("Downloading Fashion-MNIST (first run only, ~30MB — cached afterward)...")
    (x_train_full, y_train_full), (x_test_full, y_test_full) = fashion_mnist.load_data()

    # Train: first 500 per class. Validation: next 100 per class (no overlap with train).
    save_images(x_train_full, y_train_full, "train", IMAGES_PER_CLASS_TRAIN, start_offset=0)
    save_images(x_train_full, y_train_full, "validation", IMAGES_PER_CLASS_VAL, start_offset=IMAGES_PER_CLASS_TRAIN)
    save_images(x_test_full, y_test_full, "test", IMAGES_PER_CLASS_TEST, start_offset=0)


if __name__ == "__main__":
    prepare_dataset()
    print("\nDataset preparation complete.")