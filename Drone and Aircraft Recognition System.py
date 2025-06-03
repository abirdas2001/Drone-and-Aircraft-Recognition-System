import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import streamlit as st

# --------------------------------------------------------------------------

dataset_path_1 = "data"  # Dataset for aircraft
dataset_path_2 = "drone_dataset_yolo"  # Dataset for drone

data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data_1 = data_gen.flow_from_directory(dataset_path_1, target_size=(128, 128), batch_size=32, class_mode="binary", subset="training")
train_data_2 = data_gen.flow_from_directory(dataset_path_2, target_size=(128, 128), batch_size=32, class_mode="binary", subset="training")

val_data_1 = data_gen.flow_from_directory(dataset_path_1, target_size=(128, 128), batch_size=32, class_mode="binary", subset="validation")
val_data_2 = data_gen.flow_from_directory(dataset_path_2, target_size=(128, 128), batch_size=32, class_mode="binary", subset="validation")

# -------------------------------------------------------------------------------

model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # Binary classification
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_data_1, validation_data=val_data_1, epochs=5)
model.fit(train_data_2, validation_data=val_data_2, epochs=5  )

model.summary()

# -------------------------------------------------------------------------------

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0) / 255.0
    prediction = model.predict(img)[0][0]
    return "Drone" if prediction > 0.5 else "Aircraft"

def capture_from_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow("Capture Image", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("webcam_image.jpg", frame)
            break
    cap.release()
    cv2.destroyAllWindows()
    return predict_image("webcam_image.jpg")

# -------------------------------------------------------------------------------

st.title("Aircraft vs Drone Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.read())
    result = predict_image("uploaded_image.jpg")
    st.image("uploaded_image.jpg", caption="Uploaded Image", use_container_width=True)
    st.write(f"Predicted Object: {result}")

if st.button("Capture from Webcam"):
    result = capture_from_webcam()
    st.write(f"Webcam Capture Prediction: {result}")