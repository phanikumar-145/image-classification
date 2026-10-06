"""
predict.py
Loads the saved transfer learning model and predicts the class of a new image.
"""

import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "models/transfer_best.keras"
CLASS_NAMES_PATH = "models/class_names.json"
IMAGE_SIZE = (128, 128)


def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)


def load_and_preprocess_image(image_path):
    """Load an image file, resize it, and apply the same preprocessing used during training."""
    img = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # model expects a batch dimension
    img_array = preprocess_input(img_array)  # MobileNetV2's specific [-1, 1] scaling
    return img_array


def predict_image(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_names()

    img_array = load_and_preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)[0]

    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100

    print(f"Image: {image_path}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.1f}%")

    print("\nAll class probabilities:")
    for name, prob in sorted(zip(class_names, predictions), key=lambda x: -x[1]):
        print(f"  {name}: {prob * 100:.1f}%")

    return predicted_class, confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    predict_image(image_path)