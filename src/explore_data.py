"""
explore_data.py
Visualizes sample images and class distribution before training.
"""

import os
import matplotlib.pyplot as plt

from src.data_preprocessing import load_datasets

FIG_DIR = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def plot_sample_images(train_ds, class_names):
    """Show one grid of sample images with their labels."""
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            label_idx = labels[i].numpy().argmax()
            plt.title(class_names[label_idx])
            plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/sample_images.png")
    plt.close()
    print(f"Saved sample images to {FIG_DIR}/sample_images.png")


def plot_class_distribution():
    """Count images per class in train/validation/test and plot as a bar chart."""
    splits = ["train", "validation", "test"]
    counts_per_split = {}

    for split in splits:
        split_dir = f"data/{split}"
        class_counts = {}
        for class_name in sorted(os.listdir(split_dir)):
            class_path = os.path.join(split_dir, class_name)
            if os.path.isdir(class_path):
                class_counts[class_name] = len(os.listdir(class_path))
        counts_per_split[split] = class_counts

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, split in zip(axes, splits):
        counts = counts_per_split[split]
        ax.bar(counts.keys(), counts.values())
        ax.set_title(f"Class Distribution: {split}")
        ax.set_ylabel("Number of images")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/class_distribution.png")
    plt.close()
    print(f"Saved class distribution plot to {FIG_DIR}/class_distribution.png")

    for split, counts in counts_per_split.items():
        print(f"\n{split}: {counts}")


if __name__ == "__main__":
    train_ds, val_ds, test_ds, class_names = load_datasets()
    plot_sample_images(train_ds, class_names)
    plot_class_distribution()