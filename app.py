import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Pneumonia Detection")

st.title("Pneumonia Detection Using CNN")
st.write("Upload a chest X-ray image for an experimental model prediction.")

st.warning(
    "This is a college project demonstration only. "
    "It is not intended for medical diagnosis."
)

MODEL_PATH = "pneumonia_cnn_model.keras"

@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_cnn_model()
    uploaded_file = st.file_uploader(
        "Upload a chest X-ray image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Chest X-ray")

        image = image.resize((150, 150))
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        probability = float(
            model.predict(image_array, verbose=0)[0][0]
        )

        st.write(f"Model probability: {probability * 100:.2f}%")

        if probability >= 0.5:
            st.write("Model output: PNEUMONIA")
        else:
            st.write("Model output: NORMAL")

except Exception as error:
    st.error(f"Error loading model: {error}")
