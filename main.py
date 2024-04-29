import cv2
import mediapipe as mp
import numpy as np
from ObjectDetection import TeachableMachineModel

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Path to the model and label files
model_path = "tflite_model/model_unquant.tflite"
label_path = "tflite_model/labels.txt"

# Load the Teachable Machine model
model = TeachableMachineModel(model_path, label_path)

# Fungsi untuk mendeteksi gesture tangan, menampilkan skeleton, dan bounding box
def detect_hand_gesture(image):
    # Mendeteksi tangan dalam gambar
    results = hands.process(image)
    # Inisialisasi variabel untuk menyimpan koordinat bounding box
    bounding_boxes = []
    # Jika tangan terdeteksi
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Tentukan apakah tangan kanan atau kiri
            is_right_hand = handedness.classification[0].label == 'Right'

            # Gambar bounding box
            x_min, y_min, x_max, y_max = calculate_bounding_box(hand_landmarks, image.shape[1], image.shape[0])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            bounding_boxes.append((x_min, y_min, x_max, y_max))

            # Gambar skeleton tangan
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Crop gambar sesuai dengan bounding box
            cropped_image = crop_image(image, x_min, y_min, x_max, y_max)
            # Tampilkan gambar crop sesuai dengan tangan kanan atau kiri
            if cropped_image is not None:
                if is_right_hand:
                    # Make predictions
                    predictions = model.predict(cropped_image)
                    label_index = np.argmax(predictions)
                    labelModel = model.labels[label_index]
                    label = "Kanan" if is_right_hand else "Kiri"
                    cv2.putText(image, label+" "+labelModel, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    # print("Predicted label:", label)
                    cv2.imshow("Tangan kanan", cropped_image)
                else:
                    # Make predictions
                    predictions = model.predict(cropped_image)
                    label_index = np.argmax(predictions)
                    labelModel = model.labels[label_index]
                    label = "Kanan" if is_right_hand else "Kiri"
                    cv2.putText(image, label+" "+labelModel, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    print("Predicted label:", label)
                    cv2.imshow("Tangan kiri", cropped_image)
    # Crop gambar sesuai dengan bounding box tangan
    if len(bounding_boxes) == 2:
        cropped_image = crop_double_handed(image, bounding_boxes[0], bounding_boxes[1])
        if cropped_image is not None:
            cv2.imshow("Double Handed", cropped_image)

    return image

# Fungsi untuk menghitung bounding box
def calculate_bounding_box(hand_landmarks, image_width, image_height):
    x_min, x_max = image_width, 0
    y_min, y_max = image_height, 0
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * image_width), int(landmark.y * image_height)
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)
    return x_min, y_min, x_max, y_max

def crop_image(image, x_min, y_min, x_max, y_max):
    # Periksa ukuran gambar sebelum memotong
    if x_min >= 0 and y_min >= 0 and x_max <= image.shape[1] and y_max <= image.shape[0]:
        cropped_image = image[y_min:y_max, x_min:x_max]
        return cropped_image
    else:
        return None
# Fungsi untuk memotong gambar sesuai dengan bounding box tangan
def crop_double_handed(image, bounding_box1, bounding_box2):
    x1_min, y1_min, x1_max, y1_max = bounding_box1
    x2_min, y2_min, x2_max, y2_max = bounding_box2
    # Periksa apakah koordinat bounding box valid
    if x1_min >= 0 and y1_min >= 0 and x1_max <= image.shape[1] and y1_max <= image.shape[0] \
            and x2_min >= 0 and y2_min >= 0 and x2_max <= image.shape[1] and y2_max <= image.shape[0]:
        cropped_image = image[min(y1_min, y2_min):max(y1_max, y2_max), min(x1_min, x2_min):max(x1_max, x2_max)]
        return cropped_image
    else:
        return None

# Buka webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Baca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    # Deteksi gesture tangan dan tampilkan skeleton serta bounding box
    frame_with_detection = detect_hand_gesture(frame)
    
    # Tampilkan frame dengan skeleton tangan dan bounding box
    cv2.imshow('Hand Gesture with Skeleton and Bounding Box', frame_with_detection)
    
    # Hentikan program jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup webcam dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()
