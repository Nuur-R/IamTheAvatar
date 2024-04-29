import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

class TeachableMachineModel:
    def __init__(self, model_path, label_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.labels = self.load_labels(label_path)

    def load_labels(self, label_path):
        with open(label_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def preprocess_input(self, frame):
        # Resize frame to match input shape
        input_shape = self.input_details[0]['shape'][1:3]
        frame_resized = cv2.resize(frame, (input_shape[1], input_shape[0]))
        # Convert frame to RGB (if not already in RGB)
        if len(frame_resized.shape) == 2:
            frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2RGB)
        elif frame_resized.shape[2] == 1:
            frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2RGB)
        else:
            frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        # Normalize pixel values to [0, 1]
        frame_resized = frame_resized.astype(np.float32) / 255.0
        # Add batch dimension
        frame_expanded = np.expand_dims(frame_resized, axis=0)
        return frame_expanded

    def predict(self, frame):
        input_data = self.preprocess_input(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data

# # Contoh penggunaan:
# model_path = "tflite_model/model_unquant.tflite"
# label_path = "tflite_model/labels.txt"
# model = TeachableMachineModel(model_path, label_path)
# frame = cv2.imread("3.png")
# predictions = model.predict(frame)
# label_index = np.argmax(predictions)
# label = model.labels[label_index]
# print("Predicted label:", label)
