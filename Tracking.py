import cv2 as cv
import numpy as np
from keras.api._v2.keras import preprocessing
import tensorflow as tf
from tensorflow import keras
from Model import model_0, class_names  # Import โมเดลและคลาสจากไฟล์ model.py


def preprocess_frame(frame):
    # Resize the frame to fit the input size of the model
    resized_frame = cv.resize(frame, (224, 224))

    # Normalize pixel values to be in the range [0, 1]
    preprocessed_frame = resized_frame.astype('float32') / 255.0

    # Add an extra dimension to match the model's input shape
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)

    return preprocessed_frame

# Load the model
model = model_0

# Open the video
video_path = "C:\\Users\\wanwisa.j\\Documents\\GitHub\\Vehicle-classification\\archive\\Audi.mp4"
cap = cv.VideoCapture(video_path)

# Get the width and height of the video
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame (resize, normalize, etc.)
    preprocessed_frame = preprocess_frame(frame)

    # Make predictions
    predictions = model.predict(preprocessed_frame)  # ทำนายคลาสของวัตถุในเฟรม

    # Draw bounding boxes
    for prediction in predictions:  # Loop through all predictions
        for class_probs in predictions:  # Loop through all class probabilities in a prediction
            class_id = np.argmax(class_probs)  # Get the class ID with highest probability
            confidence = class_probs[class_id]  # Get the confidence of the predicted class

            # Check if the confidence is above a certain threshold
            if confidence > 0:
                # Get the coordinates and size of the bounding box
                center_x = int(class_probs[0] * width)
                center_y = int(class_probs[1] * height)
                w = int(class_probs[2] * width)
                h = int(class_probs[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw the bounding box
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color, thickness 2

                # Print the class name and confidence on the bounding box
                class_name = class_names[class_id]
                text = f"{class_name}: {confidence:.2f}"
                cv.putText(frame, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv.LINE_AA)



    # Show the frame
    cv.imshow('Frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv.destroyAllWindows()
