import cv2
import os
import datetime
import mediapipe as mp


# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

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

            # Tambahkan label
            label = "Right" if is_right_hand else "Left"
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Gambar skeleton tangan
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return image, bounding_boxes

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

# Membuat folder dataset
def create_dataset_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        os.makedirs(os.path.join(folder_name, 'Right'))
        os.makedirs(os.path.join(folder_name, 'Left'))
        os.makedirs(os.path.join(folder_name, 'DoubleHanded'))

def crop_double_handed(image, bounding_box1, bounding_box2):
    # Gabungkan kotak pembatas (bounding box) dari kedua tangan
    x1_min, y1_min, x1_max, y1_max = bounding_box1
    x2_min, y2_min, x2_max, y2_max = bounding_box2
    x_min = min(x1_min, x2_min)
    y_min = min(y1_min, y2_min)
    x_max = max(x1_max, x2_max)
    y_max = max(y1_max, y2_max)

    # Periksa apakah koordinat bounding box valid
    if x_min >= 0 and y_min >= 0 and x_max <= image.shape[1] and y_max <= image.shape[0]:
        cropped_image = image[y_min:y_max, x_min:x_max]
        return cropped_image
    else:
        return None

# Simpan gambar dengan format nama yang sesuai
def save_image(image, folder_name, bounding_boxes):
    for i, bbox in enumerate(bounding_boxes):
        x_min, y_min, x_max, y_max = bbox
        hand_type = "DoubleHanded" if len(bounding_boxes) > 1 else "Right" if bounding_boxes[0][0] > image.shape[1] / 2 else "Left"
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if hand_type == "DoubleHanded":
            # Tambahkan indeks ke nama file
            file_name = f"{hand_type}_{current_time}_{i}.png"
            # Pastikan folder DoubleHanded telah dibuat sebelumnya
            folder_path = os.path.join(folder_name, hand_type)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            # Simpan gambar double-handed
            double_handed_image = crop_double_handed(image, bounding_boxes[0], bounding_boxes[1])
            cv2.imwrite(os.path.join(folder_path, file_name), double_handed_image)
        else:
            # Tidak perlu indeks untuk tangan tunggal
            file_name = f"{hand_type}_{current_time}.png"
            folder_path = os.path.join(folder_name, hand_type)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            cv2.imwrite(os.path.join(folder_path, file_name), image[y_min:y_max, x_min:x_max])


# Buka webcam
cap = cv2.VideoCapture(0)

# Nama folder dataset
folder_name = input("Masukkan nama folder dataset: ")
create_dataset_folder(folder_name)

while cap.isOpened():
    # Baca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    # Deteksi gesture tangan dan tampilkan skeleton serta bounding box
    frame_with_detection, bounding_boxes = detect_hand_gesture(frame)
    
    # Tampilkan frame dengan skeleton tangan dan bounding box
    cv2.imshow('Hand Gesture with Skeleton and Bounding Box', frame_with_detection)
    
    # Simpan gambar saat tombol "s" ditekan
    if cv2.waitKey(1) & 0xFF == ord('s'):
        save_image(frame_with_detection, folder_name, bounding_boxes)
    
    # Hentikan program jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup webcam dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()
