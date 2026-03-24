import streamlit as st
import cv2
import numpy as np
from model.inference import generate_caption
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model

base_model = InceptionV3(weights='imagenet')
cnn_model = Model(base_model.input, base_model.layers[-2].output)

st.title("🖼 Image Caption Generator")

uploaded_file = st.file_uploader("Upload an image")

if uploaded_file:
    img = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img, 1)
    img_resized = cv2.resize(img, (299, 299))
    img_processed = preprocess_input(img_resized)
    img_processed = np.expand_dims(img_processed, axis=0)

    feature = cnn_model.predict(img_processed)
    caption = generate_caption(feature)

    st.image(img, caption="Uploaded Image")
    st.subheader("Generated Caption:")
    st.write(caption)