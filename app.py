"""
app.py
Streamlit UI for uploading an image and getting a predicted clothing class.
Run with: streamlit run app.py
"""

import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "models/transfer_best.keras"
CLASS_NAMES_PATH = "models/class_names.json"
IMAGE_SIZE = (128, 128)

st.set_page_config(page_title="Image Classifier", layout="centered")


@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    return model, class_names


def preprocess_uploaded_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array, img


st.title("Clothing Image Classifier")
st.caption("Upload an image and the model will predict which of 5 clothing categories it belongs to.")

model, class_names = load_model_and_classes()
st.info(f"Model trained to recognize: {', '.join(class_names)}")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_array, display_img = preprocess_uploaded_image(uploaded_file)

    st.image(display_img, caption="Uploaded image", width=300)

    if st.button("Predict"):
        with st.spinner("Running prediction..."):
            predictions = model.predict(img_array, verbose=0)[0]

        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = predictions[predicted_index] * 100

        st.success(f"Predicted class: **{predicted_class}**")
        st.metric("Confidence", f"{confidence:.1f}%")

        st.subheader("All class probabilities")
        sorted_results = sorted(zip(class_names, predictions), key=lambda x: -x[1])
        for name, prob in sorted_results:
            st.write(f"{name}: {prob * 100:.1f}%")
            st.progress(float(prob))