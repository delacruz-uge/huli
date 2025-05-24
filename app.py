import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the model
@st.cache_resource
def load_trained_model():
    model = tf.keras.models.load_model("bird_drone_classifier.keras")
    return model

model = load_trained_model()

# App title
st.title("Bird vs Drone Classifier")

# File uploader
uploaded_file = st.file_uploader("Upload an image of a bird or drone", type=["jpg", "jpeg", "png"])

# Prediction
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Resize and preprocess the image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    class_label = "Drone" if prediction[0][0] > 0.5 else "Bird"

    st.markdown(f"### Prediction: **{class_label}** ({prediction[0][0]:.2f})")
