import cv2
import mediapipe as mp

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Fungsi untuk mendeteksi gesture tangan, menampilkan skeleton, dan bounding box
def detect_hand_gesture(image):
    # Mendeteksi tangan dalam gambar
    results = hands.process(image)
    # Jika tangan terdeteksi
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Tentukan apakah tangan kanan atau kiri
            is_right_hand = handedness.classification[0].label == 'Right'

            # Gambar bounding box
            x_min, y_min, x_max, y_max = calculate_bounding_box(hand_landmarks, image.shape[1], image.shape[0])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Tambahkan label
            label = "Kanan" if is_right_hand else "Kiri"
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Gambar skeleton tangan
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Crop gambar sesuai dengan bounding box
            cropped_image = crop_image(image, x_min, y_min, x_max, y_max)
            # Tampilkan gambar crop sesuai dengan tangan kanan atau kiri
            if cropped_image is not None:
                if is_right_hand:
                    cv2.imshow("Tangan kanan", cropped_image)
                else:
                    cv2.imshow("Tangan kiri", cropped_image)

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

# Fungsi untuk memotong gambar sesuai dengan bounding box
def crop_image(image, x_min, y_min, x_max, y_max):
    # Periksa ukuran gambar sebelum memotong
    if x_min >= 0 and y_min >= 0 and x_max <= image.shape[1] and y_max <= image.shape[0]:
        cropped_image = image[y_min:y_max, x_min:x_max]
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
