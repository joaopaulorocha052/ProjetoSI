import cv2
import mediapipe as mp
import face_recognition
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import sys

path_modelo = 'face_landmarker.task'

if len(sys.argv) > 2:
	img_file = sys.argv[1]
	video_input = sys.argv[2]
else:
	img_file = 'pedro.jpg'
	video_input = 0

img_rosto = face_recognition.load_image_file(img_file)

encoding_rosto = face_recognition.face_encodings(img_rosto)[0]

#MediaPipe
base_options = python.BaseOptions(model_asset_path=path_modelo)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(video_input)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    timestamp_ms = int(time.time() * 1000)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    nome_exibido = "Desconhecido"
    cor_box = (0, 0, 255) #Vermelho

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]
        h, w, _ = frame.shape
        
        x_coords = [int(l.x * w) for l in landmarks]
        y_coords = [int(l.y * h) for l in landmarks]
        
        top, bottom = min(y_coords), max(y_coords)
        left, right = min(x_coords), max(x_coords)

        #reconhecimento naive
        face_location = [(top, right, bottom, left)]
        face_encodings_frame = face_recognition.face_encodings(rgb_frame, face_location)
        if face_encodings_frame:
            encoding_atual = face_encodings_frame[0]
            distancia = face_recognition.face_distance([encoding_rosto], encoding_atual)[0]
            if distancia < 0.6:
                nome_exibido = f"Pedro ({distancia:.2f})"
                cor_box = (0, 255, 0) #VERDE
        cv2.rectangle(frame, (left, top), (right, bottom), cor_box, 2)
        cv2.putText(frame, nome_exibido, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_box, 2)
    cv2.imshow('Reconhecimento Naive MediaPipe', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
