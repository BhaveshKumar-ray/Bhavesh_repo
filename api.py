from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from PIL import Image
import io

# =========================================================================
# 🛑 DEFINITIONS AND MODEL STRUCTURE FIX (THE FINAL SOLUTION)
# =========================================================================
# We will define the architecture and load weights from the best H5 file.
WEIGHTS_PATH = "traffic_sign_model_cnn.h5" 
IMG_SIZE = (64, 64)
NUM_CLASSES = 85

# Define the definitive class names list
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

# Initialize FastAPI App
app = FastAPI(title="Traffic Sign ML Service")

# Global model variable
model = None

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


# Load model upon startup
@app.on_event("startup")
def load_ml_model():
    global model
    try:
        # Create the perfect model structure with Softmax
        model = create_cnn_model(NUM_CLASSES, IMG_SIZE + (3,))
        
        # Load weights from the saved H5 file (Bypasses Keras's faulty loading of structure)
        model.load_weights(WEIGHTS_PATH)
        
        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        print(f"✅ FastAPI Model Loaded (Weights from: {WEIGHTS_PATH}) and Ready.")
    except Exception as e:
        # Force a proper server crash if loading fails
        import sys
        print(f"❌ FATAL: Model loading failed: {e}")
        sys.exit(1)


# API Endpoint for Prediction
@app.post("/predict/")
async def predict_traffic_sign(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="ML Model not loaded.")

    # 1. Read Image Data from UploadFile
    contents = await file.read()
    image_bytes = io.BytesIO(contents)
    
    try:
        # 2. Preprocess Image
        image = Image.open(image_bytes).convert("RGB")
        img_model = image.resize(IMG_SIZE)
        img_array = np.array(img_model) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image file: {e}")

    # 3. Predict
    predictions = model.predict(img_array, verbose=0) 

    # The output is correctly scaled probabilities now due to the structure definition
    pred_probs = predictions[0]
    
    predicted_index = np.argmax(pred_probs)
    confidence = np.max(pred_probs) * 100
    
    # Map index to name
    sign_name = CLASS_NAMES[predicted_index]

    # 4. Return structured JSON result
    return JSONResponse(content={
        "filename": file.filename,
        "prediction": sign_name,
        "confidence": round(confidence, 4),
        "class_index": int(predicted_index)
    })