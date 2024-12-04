import subprocess
import sys

def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError as e:
        print(f"Error while installing requirements: {e}")
        sys.exit(1)

# Run the function to install requirements
install_requirements()
import warnings
warnings.filterwarnings("ignore") 
import streamlit as st
from PIL import Image
Image.warnings.simplefilter("ignore", Image.DecompressionBombWarning)
import tensorflow as tf
import os
import numpy as np
import pandas as pd

# Paths
MODEL_PATH = r"C:\Users\LENOVO\Desktop\7sem project\traffic_sign_model_new.h5"  # Path to your saved model
CSV_PATH = r"C:\Users\LENOVO\Desktop\7sem project\Indian-Traffic Sign-Dataset\traffic_sign.csv"  # Path to your CSV file with ClassId and Name
class_labels = pd.read_csv(CSV_PATH)

# Convert to a dictionary for easy lookup
class_labels_dict = pd.Series(class_labels.Name.values, index=class_labels.ClassId).to_dict()

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    # Ensure the image is RGB (convert if it has an alpha channel)
    image = image.convert("RGB")  # Converts RGBA or grayscale to RGB
    # Resize the image to the expected input size of the model
    image = image.resize((64, 64))  # Replace 64x64 with your model's expected size
    # Convert image to a NumPy array and normalize
    image_array = np.array(image) / 255.0  
    # Add batch dimension (for batch processing in the model)
    image_array = np.expand_dims(image_array, axis=0)  
    return image_array

def predict_sign(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    class_id = np.argmax(predictions)
    confidence = predictions[0][class_id]
    
    # Get the name of the traffic sign from the CSV mapping
    sign_name = class_labels_dict.get(class_id, "Unknown Sign")
    return sign_name, confidence

st.title("Traffic Sign Classifier")
st.write("Upload an image of a traffic sign to predict its class.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    image_resized = image.resize((50, 50))
    col1, col2, col3 = st.columns([1, 4, 1])  # Columns to center the image
    with col2:
        st.image(image_resized, caption="Uploaded Image")
    # st.image(image_resized, caption="Uploaded Image")

    # Predict button
    if st.button("Predict"):
        sign_name, confidence = predict_sign(image)
        st.write(f"Predicted Traffic Sign: {sign_name}")
        st.write(f"Confidence: {confidence * 100:.2f}%")
