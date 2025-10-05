import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from PIL import Image
import io
import os

# =========================================================================
# 🛑 DEFINITIONS AND MODEL STRUCTURE (Architecture Bypass Fix)
# We load WEIGHTS ONLY into this perfectly defined structure to ensure accuracy.
# =========================================================================
# Path to your high-accuracy CNN model weights (H5 is the format saved in Colab)
# Streamlit will find this file in the root of your GitHub repo.
# Path to your high-accuracy CNN model weights (H5 is the format saved in Colab)
# Streamlit will find this file in the root of your GitHub repo.
WEIGHTS_PATH = "traffic_sign_model_cnn.h5"
IMG_SIZE = (64, 64)
NUM_CLASSES = 85

# Definitive Class Names List (Must be in correct index order)
CLASS_NAMES = [
    'ALL_MOTOR_VEHICLE_PROHIBITED', 'AXLE_LOAD_LIMIT', 'BARRIER_AHEAD', 
    'BULLOCK_AND_HANDCART_PROHIBITED', 'BULLOCK_PROHIBITED', 'CATTLE', 
    'COMPULSARY_AHEAD', 'COMPULSARY_AHEAD_OR_TURN_LEFT', 'COMPULSARY_AHEAD_OR_TURN_RIGHT', 
    'COMPULSARY_CYCLE_TRACK', 'COMPULSARY_KEEP_LEFT', 'COMPULSARY_KEEP_RIGHT', 
    'COMPULSARY_MINIMUM_SPEED', 'COMPULSARY_SOUND_HORN', 'COMPULSARY_TURN_LEFT', 
    'COMPULSARY_TURN_LEFT_AHEAD', 'COMPULSARY_TURN_RIGHT', 'COMPULSARY_TURN_RIGHT_AHEAD', 
    'CROSS_ROAD', 'CYCLE_CROSSING', 'CYCLE_PROHIBITED', 
    'DANGEROUS_DIP', 'DIRECTION', 'FALLING_ROCKS', 
    'FERRY', 'GAP_IN_MEDIAN', 'GIVE_WAY', 
    'GUARDED_LEVEL_CROSSING', 'HANDCART_PROHIBITED', 'HEIGHT_LIMIT', 
    'HORN_PROHIBITED', 'HUMP_OR_ROUGH_ROAD', 'LEFT_HAIR_PIN_BEND', 
    'LEFT_HAND_CURVE', 'LEFT_REVERSE_BEND', 'LEFT_TURN_PROHIBITED', 
    'LENGTH_LIMIT', 'LOAD_LIMIT', 'LOOSE_GRAVEL', 
    'MEN_AT_WORK',  
    'NARROW_BRIDGE', 'NARROW_ROAD_AHEAD', 'NO_ENTRY', 
    'NO_PARKING', 
    'NO_STOPPING_OR_STANDING', 'OVERTAKING_PROHIBITED', 
    'PASS_EITHER_SIDE', 'PEDESTRIAN_CROSSING', 'PEDESTRIAN_PROHIBITED', 
    'PRIORITY_FOR_ONCOMING_VEHICLES', 'QUAY_SIDE_OR_RIVER_BANK', 'RESTRICTION_ENDS', 
    'RIGHT_HAIR_PIN_BEND', 'RIGHT_HAND_CURVE', 'RIGHT_REVERSE_BEND', 
    'RIGHT_TURN_PROHIBITED', 'ROAD_WIDENS_AHEAD', 'ROUNDABOUT', 
    'SCHOOL_AHEAD', 'SIDE_ROAD_LEFT', 'SIDE_ROAD_RIGHT', 
    'SLIPPERY_ROAD', 'SPEED_LIMIT_15', 'SPEED_LIMIT_20', 
    'SPEED_LIMIT_30', 'SPEED_LIMIT_40', 'SPEED_LIMIT_5', 
    'SPEED_LIMIT_50', 'SPEED_LIMIT_60', 'SPEED_LIMIT_70', 
    'SPEED_LIMIT_80', 'STAGGERED_INTERSECTION', 'STEEP_ASCENT', 
    'STEEP_DESCENT', 'STOP', 
    'STRAIGHT_PROHIBITED', 'TONGA_PROHIBITED', 'TRAFFIC_SIGNAL', 
    'TRUCK_PROHIBITED', 'TURN_RIGHT', 'T_INTERSECTION', 
    'UNGUARDED_LEVEL_CROSSING', 'U_TURN_PROHIBITED', 'WIDTH_LIMIT', 
    'Y_INTERSECTION' 
] 

# 1. Define the model structure exactly as trained in Colab
def create_cnn_model(num_classes, input_shape):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax') # Softmax is CORRECTLY defined here
    ])
    return model

# 2. Function to load the model just once and cache it
@st.cache_resource 
def load_and_init_model(path):
    try:
        # Create the perfect model structure with Softmax
        model = create_cnn_model(NUM_CLASSES, IMG_SIZE + (3,))
        
        # Load weights from the saved H5 file
        model.load_weights(path)
        
        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"❌ FATAL: Model loading failed. Ensure '{path}' exists and is uncorrupted.")
        st.code(f"Error Details: {e}")
        return None

# Load the model upon app start
model = load_and_init_model(WEIGHTS_PATH)

# ----------------- STREAMLIT FRONTEND -----------------

st.markdown("# 🚦 Indian Traffic Sign Recognition System")
st.markdown("### **Project Demo: Utilizing $79.50\%$ Accurate CNN Service**")

if model is None:
    st.stop() # Stop execution if the model failed to load

uploaded_file = st.file_uploader("Upload an image of a Traffic Sign:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and display the image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        img_model = image.resize(IMG_SIZE)
        img_array = np.array(img_model) / 255.0
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

        # Predict
        predictions = model.predict(img_array, verbose=0)
        
        # Output is correctly scaled probabilities
        pred_probs = predictions[0] 
        predicted_index = np.argmax(pred_probs)
        confidence = np.max(pred_probs) * 100
        sign_name = CLASS_NAMES[predicted_index]

        # Display results with dynamic styling
        st.subheader("Classification Result:")
        
        if confidence > 85:
             st.success(f"✅ HIGH CONFIDENCE: **{sign_name}**")
        elif confidence > 65:
             st.warning(f"⚠️ GOOD CONFIDENCE: **{sign_name}**")
        else:
             st.error(f"❌ LOW CONFIDENCE: **{sign_name}**")
        
        st.metric(label="Predicted Sign", value=sign_name)
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
        st.caption(f"Predicted Index: {predicted_index}")
        
        # Displaying top 5 predictions for deeper analysis
        top_k = 5
        top_indices = np.argsort(pred_probs)[::-1][:top_k]
        top_confidences = np.sort(pred_probs)[::-1][:top_k] * 100

        st.markdown("**Top 5 Predictions:**")
        for i in range(top_k):
            st.text(f"  {i+1}. {CLASS_NAMES[top_indices[i]]}: {top_confidences[i]:.2f}%")

    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")
