from ObjectDetection import TeachableMachineModel
import numpy as np
import cv2

# Path to the model and label files
model_path = "tflite_model/model_unquant.tflite"
label_path = "tflite_model/labels.txt"

# Load the Teachable Machine model
model = TeachableMachineModel(model_path, label_path)

# Read an image
image_path = "1.png"
frame = cv2.imread(image_path)

# Make predictions
predictions = model.predict(frame)

# Process predictions as needed
label_index = np.argmax(predictions)
label = model.labels[label_index]
print("Predicted label:", label)
