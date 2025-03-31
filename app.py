from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import os

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model
model_path = os.path.join(current_dir, "model", "keras_model.h5")
model = load_model(model_path, compile=False)

# Load the labels
labels_path = os.path.join(current_dir, "model", "labels.txt")
class_names = open(labels_path, "r", encoding='utf-8').readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(1)

# Set window size (1.5 times larger than original 224x224)
window_width = 336  # 224 * 1.5
window_height = 336  # 224 * 1.5

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels for model
    model_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Resize the display image to be 1.5 times larger
    display_image = cv2.resize(image, (window_width, window_height), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the models input shape.
    model_image = np.asarray(model_image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    model_image = (model_image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(model_image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Prepare text to display
    text = f"Class: {class_name[2:]} Confidence: {np.round(confidence_score * 100)}%"
    
    # Add text to the image with smaller font size
    cv2.putText(display_image, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show the image in a window
    cv2.imshow("Webcam Image", display_image)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows() 