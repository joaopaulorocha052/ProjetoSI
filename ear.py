import cv2
import mediapipe as mp
import face_recognition
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.spatial import distance
import time
import sys

path_modelo = 'face_landmarker.task'

if len(sys.argv) > 2:
	arquivo_rosto = sys.argv[1]
	video_input = sys.argv[2]
else:
	arquivo_rosto = 'pedro.jpg'
	video_input =  0

#Ajustar
EAR_THRESHOLD = 0.4  # Limiar para considerar olho aberto

#Var
contador = 0
piscadas = 0

img_pedro = face_recognition.load_image_file(arquivo_rosto)
encoding_pedro = face_recognition.face_encodings(img_pedro)[0]

#Mediapipe
base_options = python.BaseOptions(model_asset_path=path_modelo)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

#Indices MediaPipe
L_EYE_TOP_BOT = [159, 145]
L_EYE_LEFT_RIGHT = [33, 133]
R_EYE_TOP_BOT = [386, 374]
R_EYE_LEFT_RIGHT = [362, 263]

def calcula_ear(landmarks, top_bot, esq_dir):
    v = distance.euclidean((landmarks[top_bot[0]].x, landmarks[top_bot[0]].y),(landmarks[top_bot[1]].x, landmarks[top_bot[1]].y))
    h = distance.euclidean((landmarks[esq_dir[0]].x, landmarks[esq_dir[1]].y),(landmarks[esq_dir[1]].x, landmarks[esq_dir[1]].y))
    return v/h

cap = cv2.VideoCapture(video_input)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h_frame, w_frame, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    timestamp = int(time.time() * 1000)
    result = detector.detect_for_video(mp_image, timestamp)

    status_liveness = "Nao Liveness"
    cor_liveness = (255, 255, 255)
    nome_identificado = ""
    if result.face_landmarks:
        landmarks = result.face_landmarks[0]
        #calculo ear
        ear_esq = calcula_ear(landmarks, L_EYE_TOP_BOT, L_EYE_LEFT_RIGHT)
        ear_dir = calcula_ear(landmarks, R_EYE_TOP_BOT, R_EYE_LEFT_RIGHT)
        ear_medio = (ear_esq + ear_dir) / 2.0
        print(ear_medio)

        if ear_medio < EAR_THRESHOLD:
            if contador != 0:
                piscadas += 1
            contador = 0
            status_liveness = "Piscou"
            cor_liveness = (0, 255, 255) # Amarelo
        else:
            contador = 1
            if piscadas > 0:
                status_liveness = "Liveness OK. Piscadas:{}".format(piscadas)
            cor_liveness = (0, 255, 0) # Verde

            xs = [int(l.x * w_frame) for l in landmarks]
            ys = [int(l.y * h_frame) for l in landmarks]
            top, right, bottom, left = min(ys), max(xs), max(ys), min(xs)

            loc = [(max(0, top-20), min(w_frame, right+20), min(h_frame, bottom+20), max(0, left-20))]
            
            encodings_frame = face_recognition.face_encodings(rgb_frame, loc)
            if encodings_frame:
                dist = face_recognition.face_distance([encoding_pedro], encodings_frame[0])[0]
                if dist < 0.6:
                    nome_identificado = f"PEDRO ({dist:.2f})"
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                else:
                    nome_identificado = "DESCONHECIDO"
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        #pontos olhos
        for i in L_EYE_TOP_BOT + R_EYE_TOP_BOT:
            p = landmarks[i]
            cv2.circle(frame, (int(p.x * w_frame), int(p.y * h_frame)), 2, cor_liveness, -1)
    else:
        status_liveness = "Nao Liveness"
        piscadas = 0
        contador = 0
        nome_identificado = ""
        cor_liveness = (255, 255, 255)
    #Exibe
    cv2.putText(frame, status_liveness, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_liveness, 2)
    if nome_identificado:
        cv2.putText(frame, nome_identificado, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Liveness EAR + Reconhecimento Naive', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
