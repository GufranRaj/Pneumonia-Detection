import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

st.set_page_config(page_title="Pneumonia Detection")

st.title(“Pneumonia Detection Using CNN”)
st.write(“Upload a chest X-ray image for an experimental model prediction.”)

st.warning(
“This is a college project demonstration only. “
“It is not intended for medical diagnosis.”
)

MODEL_PATH = “pneumonia_cnn_model.keras”
FILE_ID = “1yh75cU0TMw2UiM3QUeUDXT3g1OxycuqM”

@st.cache_resource
def load_model():
if not os.path.exists(MODEL_PATH):
gdown.download(
f”https://drive.google.com/uc?id={FILE_ID}”,
MODEL_PATH,
quiet=False
)

return tf.keras.models.load_model(MODEL_PATH)

uploaded_file = st.file_uploader(
“Choose a chest X-ray image”,
type=[“jpg”, “jpeg”, “png”]
)

if uploaded_file is not None:
image = Image.open(uploaded_file).convert(“RGB”)
st.image(image, caption=“Uploaded X-ray”, use_container_width=True)

if st.button("Analyze X-ray"):
    with st.spinner("Analyzing image..."):
        model = load_model()
        image = image.resize((150, 150))
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        probability = float(
            model.predict(image_array, verbose=0)[0][0]
        )
    st.subheader("Experimental Model Result")
    st.write(f"Pneumonia score: {probability * 100:.2f}%")
    if probability >= 0.5:
        st.write("Model output: PNEUMONIA")
    else:
        st.write("Model output: NORMAL")
    st.caption(
        "This model may produce incorrect results. "
        "Do not use this output for medical decisions."
    )
  if st.button("Analyze X-ray"):
    with st.spinner("Analyzing image..."):
        model = load_model()

        image = image.resize((150, 150))
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        probability = float(
            model.predict(image_array, verbose=0)[0][0]
        )

    st.subheader("Experimental Model Result")
    st.write(f"Pneumonia score: {probability * 100:.2f}%")

    if probability >= 0.5:
        st.write("Model output: PNEUMONIA")
    else:
        st.write("Model output: NORMAL")

    st.caption(
        "This model may produce incorrect results. "
        "Do not use this output for medical decisions."
    )
