import streamlit as st
from PIL import Image
from model import predict_image

# Streamlit GUI
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾", layout="centered")

st.title("🐾 Cat vs Dog Image Classifier (SVM)")
st.markdown("Upload an image of a **Cat or Dog**, and the model will predict which it is.")

# Upload file
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)


    with st.spinner("Predicting..."):
        label = predict_image(uploaded_file)
        st.success(f"Prediction: **{label}**")
