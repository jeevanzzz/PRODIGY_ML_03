# Prodigy Info Tech -🐾Cat vs Dog Image Classifier Using Support Vector Machine (SVM)

As part of my Virtual Machine Learning Internship, I successfully completed a project on Cat vs Dog Image Classification using Support Vector Machine (SVM). The goal of this task was to build an intuitive and interactive web application using Streamlit that classifies uploaded pet images as either a Cat or a Dog.

The backend classification model was trained on labeled image data using the Support Vector Machine (SVM) algorithm, after applying essential preprocessing steps such as resizing, flattening, and scaling. The trained model and scaler were serialized using Pickle, and the user interface was developed using Python and Streamlit to allow real-time image upload and prediction.

This project provided hands-on experience in computer vision, traditional machine learning, and deploying models through a simple and clean web UI.

## ✅ Features
**📸 Image Upload:**
Allows users to upload .jpg, .jpeg, or .png images directly through the Streamlit interface.

**🧠 Machine Learning Model:**
Utilizes a Support Vector Machine (SVM) trained on preprocessed image data to classify pets.

**🧼 Image Preprocessing:**
Uploaded images are resized,flattened,scaled to match the model’s training specifications for accurate predictions.

**⚖️ Model & Scaler Serialization:**
Uses Pickle to load pre-trained SVM model and scaler for quick, on-demand predictions.

**🐾 Real-time Prediction Output:**
Displays a label ("Cat" or "Dog") based on the model’s prediction instantly after image upload.

**🌐 Web Interface with Streamlit:**
Built using Streamlit to create a clean, responsive, and easy-to-use user interface.

**💡 Lightweight & Fast:**
Runs locally with minimal system requirements and delivers predictions almost instantly.


## 🚀 Demo
Upload an image, and the model will predict whether it's a cat or a dog.
![image](https://github.com/user-attachments/assets/3f099e39-4533-4a70-a4cf-5e6ede4cfc63)

## 🧠 Tech Stack
- Python 🐍
- Scikit-learn
- NumPy
- Pillow (PIL)
- Streamlit

## 🧪 How It Works
- Upload an image (.jpg, .jpeg, .png)
- The image is resized to 124x124, flattened, and scaled
- The SVM model makes a prediction
- The result (Cat or Dog) is displayed

📊 Dataset
Used Kaggle’s Dataset :- https://www.kaggle.com/c/dogs-vs-cats/data

📬 Contact
Feel free to connect with me on LinkedIn or drop a ⭐ if you find this project useful!
🔗 Author: [JEEVAN BANGERA] 📧 Contact: [jeevanbangera794@gamil.com]
