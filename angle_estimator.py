import math
import time

import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)  #(r"C:\Users\HP\OneDrive\Pictures\Camera Roll\WIN_20260415_17_15_38_Pro.mp4")

# Prepare video writer for saving processed output
output_path = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
        fps = 30.0
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)


def rotation_matrix_to_angles(rotation_matrix):
    """
    Calculate Euler angles from rotation matrix.
    :param rotation_matrix: A 3*3 matrix with the following structure
    [Cosz*Cosy  Cosz*Siny*Sinx - Sinz*Cosx  Cosz*Siny*Cosx + Sinz*Sinx]
    [Sinz*Cosy  Sinz*Siny*Sinx + Sinz*Cosx  Sinz*Siny*Cosx - Cosz*Sinx]
    [  -Siny             CosySinx                   Cosy*Cosx         ]
    :return: Angles in degrees for each axis
    """
    x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 +
                                                     rotation_matrix[1, 0] ** 2))
    z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.array([x, y, z]) * 180. / math.pi


while cap.isOpened():
    start_time = time.time()
    
    success, image = cap.read()
    if not success:
        break

    # Convert the color space from BGR to RGB and get Mediapipe results
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Convert the color space from RGB to BGR to display well with Opencv
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    face_coordination_in_real_world = np.array([
        [285, 528, 200],
        [285, 371, 152],
        [197, 574, 128],
        [173, 425, 108],
        [360, 574, 128],
        [391, 425, 108]
    ], dtype=np.float64)

    h, w, _ = image.shape
    face_coordination_in_image = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [1, 9, 57, 130, 287, 359]:
                    x, y = int(lm.x * w), int(lm.y * h)
                    face_coordination_in_image.append([x, y])

            face_coordination_in_image = np.array(face_coordination_in_image,
                                                  dtype=np.float64)

            # The camera matrix
            focal_length = 1 * w
            cam_matrix = np.array([[focal_length, 0, w / 2],
                                   [0, focal_length, h / 2],
                                   [0, 0, 1]])

            # The Distance Matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Use solvePnP function to get rotation vector
            success, rotation_vec, transition_vec = cv2.solvePnP(
                face_coordination_in_real_world, face_coordination_in_image,
                cam_matrix, dist_matrix)

            # Use Rodrigues function to convert rotation vector to matrix
            rotation_matrix, jacobian = cv2.Rodrigues(rotation_vec)

            result = rotation_matrix_to_angles(rotation_matrix)
            for i, info in enumerate(zip(('pitch', 'yaw', 'roll'), result)):
                k, v = info
                text = f'{k}: {int(v)}'
                cv2.putText(image, text, (20, i*30 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)


    """""Menampilkan waktu yang dibutuhkan untuk memproses setiap frame dan menyesuaikan waktu tunggu agar sesuai dengan FPS video"""""
    cv2.putText(image, f'time: {start_time:.2f}s', (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    elapsed_time = time.time() - start_time
    time_to_wait = max(0, (1.0 / fps) - elapsed_time)

    cv2.putText(image, f'elapsed time: {elapsed_time:.2f}s', (20, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.putText(image, f'time to wait: {time_to_wait:.2f}s', (20, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    

    out.write(image)

    # Draw center lines
    cv2.line(image, (w//2, 0), (w//2, h), (0, 255, 0), 2)
    cv2.line(image, (0, h//2), (w, h//2), (0, 255, 0), 2)

    cv2.imshow('Head Pose Angles', image)

         
    time.sleep(time_to_wait)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()

