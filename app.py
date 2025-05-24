import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np

# Load the trained model - use try-except for better error handling
try:
    model = load_model('bird_drone_classifier_v3.keras')
    # model = load_model('bird_drone_classifier.h5')  # Alternative if you saved as .h5
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Bird vs Drone Image Classifier")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

target_size = 180  # Should match your training/preprocessing size

def preprocess_image(image):
    """Match the preprocessing done during training"""
    # Convert to array and normalize to [0, 255]
    img_array = np.array(image) * 255.0
    
    # Resize while maintaining aspect ratio and pad
    height, width = image.size[1], image.size[0]
    scale = target_size / max(height, width)
    new_height, new_width = int(height * scale), int(width * scale)
    img_resized = image.resize((new_width, new_height))
    
    # Create blank square image
    img_padded = Image.new("RGB", (target_size, target_size))
    offset = ((target_size - new_width) // 2, (target_size - new_height) // 2)
    img_padded.paste(img_resized, offset)
    
    # Convert to array and apply MobileNetV2 preprocessing
    img_array = np.array(img_padded)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess exactly like during training
        img_array = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(img_array)[0][0]
        label = "Drone" if prediction > 0.5 else "Bird"
        confidence = max(prediction, 1 - prediction)  # Get the higher confidence
        
        st.success(f"Prediction: **{label}** (confidence: {confidence:.2%})")
        
    except Exception as e:
        st.error(f"Error processing image: {e}")