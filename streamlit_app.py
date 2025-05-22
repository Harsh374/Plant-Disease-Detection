import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import time

# Constants
IMAGE_SIZE = 224

# Title and description
st.set_page_config(page_title="Plant Disease Detection", layout="wide")
st.title("Plant Disease Detection App")
st.markdown("""
This application detects diseases in plant leaves using a Convolutional Neural Network.
Upload an image of a plant leaf, and the model will predict if it's healthy or what disease it might have.
""")

# Sidebar with information
st.sidebar.title("About")
st.sidebar.info("""
This app uses a CNN model trained on the PlantVillage dataset to identify
plant diseases from leaf images. The model can identify the following classes:
- Apple: Apple Scab
- Apple: Black Rot
- Apple: Cedar Apple Rust
- Apple: Healthy
- Tomato: Healthy
""")

# Function to load the trained model
@st.cache_resource
def load_trained_model():
    try:
        # Try to load the model
        model = load_model('plant_disease_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Convert PIL Image to numpy array
    img_array = np.array(image.convert('RGB'))
    
    # Resize image
    img_resized = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Normalize pixel values
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

# Function to make prediction
def predict(model, img_batch):
    # Class names (must match the order used during training)
    class_names = [
        'Apple___Apple_scab',
        'Apple___Black_rot', 
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Tomato___healthy'
    ]
    
    # Make prediction
    prediction = model.predict(img_batch)[0]
    
    # Get the predicted class and confidence
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    confidence = prediction[predicted_class_index] * 100
    
    # Format the class name for display (remove underscores and split by ___)
    display_name = predicted_class.replace('___', ': ').replace('_', ' ')
    
    # Get top 3 predictions for the bar chart
    top_indices = np.argsort(prediction)[-3:][::-1]
    top_classes = [class_names[i].replace('___', ': ').replace('_', ' ') for i in top_indices]
    top_confidences = [prediction[i] * 100 for i in top_indices]
    
    return display_name, confidence, top_classes, top_confidences

# Load the model
model = load_trained_model()

# Main app layout
col1, col2 = st.columns([1, 1])

with col1:
    # Upload image
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Check if model is loaded
        if model is not None:
            # Add predict button
            if st.button("Predict Disease"):
                with st.spinner("Analyzing image..."):
                    # Preprocessing
                    img_batch = preprocess_image(image)
                    
                    # Add a slight delay to show the spinner (optional)
                    time.sleep(1)
                    
                    # Make prediction
                    display_name, confidence, top_classes, top_confidences = predict(model, img_batch)
                    
                    # Display result
                    st.success(f"Prediction: **{display_name}**")
                    st.success(f"Confidence: **{confidence:.2f}%**")

with col2:
    # This column will display the prediction results
    if uploaded_file is not None and model is not None and st.session_state.get('predict_clicked', False):
        # Create a bar chart of top 3 predictions
        fig, ax = plt.subplots(figsize=(8, 5))
        y_pos = np.arange(len(top_classes))
        
        bars = ax.barh(y_pos, top_confidences, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_classes)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Confidence (%)')
        ax.set_title('Top Predictions')
        
        # Add confidence values on the bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'{top_confidences[i]:.2f}%', 
                    ha='left', va='center')
            
        st.pyplot(fig)
        
        # Show disease information
        st.subheader("Disease Information")
        
        # Add information about the detected disease
        if "Apple_scab" in display_name:
            st.markdown("""
            **Apple Scab**
            
            *Caused by*: Fungus Venturia inaequalis
            
            *Symptoms*: Dark olive-green to brown spots on leaves and fruit. Severely infected leaves may turn yellow and drop.
            
            *Treatment*: 
            - Apply fungicides early in the growing season
            - Rake and destroy fallen leaves
            - Prune trees to improve air circulation
            """)
        elif "Black_rot" in display_name:
            st.markdown("""
            **Apple Black Rot**
            
            *Caused by*: Fungus Botryosphaeria obtusa
            
            *Symptoms*: Purple flecks that develop into brown lesions, rotting fruit with concentric rings of spore-producing bodies.
            
            *Treatment*: 
            - Remove infected plant parts
            - Apply fungicides during the growing season
            - Maintain proper tree vigor through fertilization
            """)
        elif "Cedar_apple_rust" in display_name:
            st.markdown("""
            **Cedar Apple Rust**
            
            *Caused by*: Fungus Gymnosporangium juniperi-virginianae
            
            *Symptoms*: Bright orange-yellow spots on leaves and fruits, with small black dots in the center of the spots.
            
            *Treatment*: 
            - Plant resistant apple varieties
            - Remove nearby cedar trees if possible
            - Apply fungicides in early spring
            """)
        elif "healthy" in display_name:
            st.markdown("""
            **Healthy Plant**
            
            The leaf appears to be healthy with no visible signs of disease.
            
            *Maintenance Tips*:
            - Continue regular watering and fertilization
            - Monitor for early signs of pests or disease
            - Maintain good air circulation around plants
            """)

# Instructions and help section
st.markdown("---")
st.header("How to Use This App")
st.markdown("""
1. Upload a clear image of a plant leaf using the file uploader
2. Click the "Predict Disease" button
3. View the prediction results and confidence score
4. Read about the detected disease and recommended treatments

For best results:
- Use well-lit, clear images
- Focus on a single leaf
- Ensure the leaf fills most of the image
- Avoid shadows or reflections if possible
""")

# Footer
st.markdown("---")
st.caption("Plant Disease Detection App | Powered by TensorFlow & Streamlit")
