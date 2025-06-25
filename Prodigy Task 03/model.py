import pickle
import numpy as np
from PIL import Image

# Load trained model and scaler
with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((124, 124))  # Resize to match training
    image_array = np.array(image).flatten().reshape(1, -1)  # Flatten
    image_array_scaled = scaler.transform(image_array)
    return image_array_scaled

def predict_image(uploaded_file):
    processed_image = preprocess_image(uploaded_file)
    prediction = model.predict(processed_image)[0]
    label = "Cat" if prediction == 0 else "Dog"
    return label
