import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

st.set_page_config(page_title=“Pneumonia Detection”)
st.title(“Pneumonia Detection Using CNN”)

st.warning(“College project demonstration only. Not intended for medical diagnosis.”)

st.write(“Upload a chest X-ray image for an experimental model prediction.”)

MODEL_PATH = “pneumonia_cnn_model.keras”
FILE_ID = “1yh75cU0TMw2UiM3QUeUDXT3g1OxycuqM”

@st.cache_resource
def load_cnn_model():
if not os.path.exists(MODEL_PATH):
url = f”https://drive.google.com/uc?id={FILE_ID}”
gdown.download(url, MODEL_PATH, quiet=False)

return tf.keras.models.load_model(MODEL_PATH)

try:
model = load_cnn_model()
st.success(“CNN model loaded successfully!”)
except Exception as e:
st.error(f”Error loading model: {e}”)
st.stop()

uploaded_file = st.file_uploader(
“Upload a chest X-ray image”,
type=[“jpg”, “jpeg”, “png”]
)

if uploaded_file is not None:
image = Image.open(uploaded_file).convert(“RGB”)
st.image(image, caption=“Uploaded Chest X-ray”, use_container_width=True)

resized_image = image.resize((150, 150))
image_array = np.array(resized_image, dtype=np.float32) / 255.0
image_array = np.expand_dims(image_array, axis=0)
if st.button("Predict"):
    probability = float(model.predict(image_array, verbose=0)[0][0])
    st.write(f"Model pneumonia score: {probability * 100:.2f}%")
    if probability >= 0.5:
        st.write("Model output: PNEUMONIA")
    else:
        st.write("Model output: NORMAL")
    st.info("This is an experimental college project, not a medical diagnosis.")
