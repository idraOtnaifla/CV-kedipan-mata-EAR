import cv2
import mediapipe as mp
import math
import time

import serial

ser = serial.Serial(
    port='COM3',      # sesuaikan
    baudrate=9600,
    timeout=1
)

time.sleep(2)  # tunggu Arduino reset

# Inisialisasi
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                   refine_landmarks=True, min_detection_confidence=0.5)

# Parameter
EAR_THRESHOLD = 0.21
CONSEC_FRAMES = 3
COMMAND_TIMEOUT = 2  # detik

# Landmark mata
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# State
frame_counter = 0
blink_burst_count = 0
last_blink_time = time.time()

def eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):
    p = [(int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)) for i in eye_indices]
    A = math.dist(p[1], p[5])
    B = math.dist(p[2], p[4])
    C = math.dist(p[0], p[3])
    ear = (A + B) / (2.0 * C)
    return ear, p

def execute_command(n):

    commands = {
        1: "Lampu Menyala",
        2: "Lampu Mati",
        3: "Kipas Menyala",
        4: "Kipas Kecepatan 2",
        5: "Kipas Kecepatan 3",
        6: "Kipas Mati"
    }

    if n in commands:

        packet = f"#A{n}$"

        ser.write(packet.encode())

        print(f"[SEND] {packet}")
        print(f"[CMD ] {commands[n]}")

    else:
        print(f"[ERROR] Command {n} tidak valid")

# Kamera
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    current_time = time.time()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear, left_pts = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, w, h)
            right_ear, right_pts = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0

            # Gambar titik mata
            for (x, y) in left_pts + right_pts:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Hitung jeda dan eksekusi command jika perlu
            if blink_burst_count > 0 and current_time - last_blink_time > COMMAND_TIMEOUT:
                execute_command(blink_burst_count)
                blink_burst_count = 0

            # Deteksi kedipan
            if ear < EAR_THRESHOLD:
                frame_counter += 1
            else:
                if frame_counter >= CONSEC_FRAMES:
                    blink_burst_count += 1
                    last_blink_time = current_time
                frame_counter = 0

            # Tampilkan status di layar
            cv2.putText(frame, f'EAR: {ear:.2f}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f'Kedipan: {blink_burst_count}', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('Deteksi Kedipan + Perintah', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()