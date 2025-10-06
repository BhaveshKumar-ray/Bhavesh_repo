import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Activation
from tensorflow.nn import softmax # Still useful, but primary fix is the wrapper

# =========================================================================
# 🛑 ACTION 1: MODEL PATH (Loading the correct CNN model file)
# This path is confirmed to exist based on your last screenshot.
# =========================================================================
MODEL_PATH = r"C:\Users\HP\Documents\OneDrive\Desktop\TrafficSignApp\traffic_sign_model_cnn.h5"

# =========================================================================
# DEFINITIVE CLASS MAPPING (From your notebook's train_generator output)
# =========================================================================
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
    'STEEP_DESCENT', 'STOP', # Index 74 (The correct class for the image)
    'STRAIGHT_PROHIBITED', 'TONGA_PROHIBITED', 'TRAFFIC_SIGNAL', 
    'TRUCK_PROHIBITED', 'TURN_RIGHT', 'T_INTERSECTION', 
    'UNGUARDED_LEVEL_CROSSING', 'U_TURN_PROHIBITED', 'WIDTH_LIMIT', 
    'Y_INTERSECTION' 
] 
# =========================================================================

# Global model variable
model = None
try:
    if tf.io.gfile.exists(MODEL_PATH):
        # 1. Load the base model (which outputs logits)
        base_model = load_model(MODEL_PATH)

        # 2. 🧠 WRAP THE MODEL: Create a sequential model that uses the loaded model
        #    and immediately adds a Softmax layer to scale the output to probabilities.
        model = Sequential([
            base_model,
            Activation('softmax', name='softmax_output')
        ])
        
        print(f"✅ Successfully loaded and wrapped model from: {MODEL_PATH}")
    else:
        print(f"❌ ERROR: Model file not found at: {MODEL_PATH}")
except Exception as e:
    print("❌ ERROR: Failed to load model.")
    print(f"Detailed Error: {e}")

# Create main window
root = tk.Tk()
root.title("🚦 Traffic Sign Recognition App 🚦")
root.geometry("650x650")
root.configure(bg="#1E1E2E")

# Title
title_label = tk.Label(root, text="Traffic Sign Recognition System",
                       font=("Arial", 20, "bold"), fg="#00FFFF", bg="#1E1E2E")
title_label.pack(pady=15)

# Main image display
img_label = tk.Label(root, bg="#1E1E2E")
img_label.pack(pady=10)

# Prediction result
result_label = tk.Label(root, text="Upload an image to classify",
                        font=("Arial", 14, "bold"), fg="#FFD700", bg="#1E1E2E")
result_label.pack(pady=10)

# Frame for thumbnail + confidence bar
status_frame = tk.Frame(root, bg="#1E1E2E")
status_frame.pack(pady=10)

# Thumbnail
thumb_label = tk.Label(status_frame, bg="#1E1E2E")
thumb_label.pack(side="left", padx=10)

# Confidence bar canvas
conf_canvas = tk.Canvas(status_frame, width=300, height=25, bg="#2E2E3E", bd=0, highlightthickness=0)
conf_canvas.pack(side="left", padx=10)
conf_bar = conf_canvas.create_rectangle(0, 0, 0, 25, fill="#00FF7F")

# Update confidence bar dynamically
def update_confidence_bar(conf):
    conf_canvas.coords(conf_bar, 0, 0, int(3*conf), 25)  # 300px max width
    if conf >= 75:
        color = "#00FF7F"  # Green
    elif conf >= 40:
        color = "#FFD700"  # Yellow
    else:
        color = "#FF4500"  # Red
    conf_canvas.itemconfig(conf_bar, fill=color)
    conf_canvas.update()

# Upload & Predict function
def upload_image():
    if model is None:
        result_label.config(text="Model is not loaded. Cannot predict. Check console for error.", fg="#FF4500")
        update_confidence_bar(0)
        return
        
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Load and preprocess image
    img = Image.open(file_path).convert("RGB")
    
    # Display image setup (for GUI)
    img_ratio = min(300/img.width, 300/img.height)
    display_size = (int(img.width*img_ratio), int(img.height*img_ratio))
    img_display = img.resize(display_size)
    img_tk = ImageTk.PhotoImage(img_display)
    img_label.config(image=img_tk)
    img_label.image = img_tk

    # Thumbnail (50x50)
    thumb_img = img.resize((50, 50))
    thumb_tk = ImageTk.PhotoImage(thumb_img)
    thumb_label.config(image=thumb_tk)
    thumb_label.image = thumb_tk

    # Preprocess for model (64x64)
    img_model = img.resize((64, 64))
    img_array = np.array(img_model) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    # The 'model' variable is now the wrapped model with the final Softmax layer.
    predictions = model.predict(img_array, verbose=0) 
    
    # The output is already probabilities, no need for softmax() here.
    pred_probs = predictions[0] 
    predicted_index = np.argmax(pred_probs)
    confidence = np.max(pred_probs) * 100
    
    # Map index to name
    if 0 <= predicted_index < len(CLASS_NAMES):
        sign_name = CLASS_NAMES[predicted_index]
    else:
        sign_name = f"Unknown Class Index {predicted_index}"

    # Display result
    result_label.config(
        text=f"Prediction: {sign_name}\nConfidence: {confidence:.2f}%",
        fg="#00FF7F" if confidence > 75 else ("#FFD700" if confidence > 40 else "#FF4500"),
        font=("Arial", 14, "bold")
    )

    # Update confidence bar
    update_confidence_bar(confidence)

# Upload button
upload_btn = tk.Button(root, text="📂 Upload Image", command=upload_image,
                       font=("Arial", 14, "bold"), bg="#FF4500", fg="white",
                       activebackground="#FF6347", activeforeground="white",
                       relief="flat", padx=15, pady=5)
upload_btn.pack(pady=20)

# Run app
root.mainloop()