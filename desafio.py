import cv2
import mediapipe as mp
import face_recognition
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.spatial import distance
import time
import random
import sys

path_modelo = 'face_landmarker.task'

if len(sys.argv) > 2:
	arquivo_rosto = sys.argv[1]
	video_input = sys.argv[2]
else:
	arquivo_rosto = 'pedro.jpg'
	video_input = 0

#Ajustar
EAR_THRESHOLD = 0.18  # Olho fechado
MAR_THRESHOLD = 0.12  # Boca aberta/sorriso
YAW_THRESHOLD = 0.15  # Sensibilidade do giro de cabeça

#Var
testes = ["PISCAR OLHO ESQ", "PISCAR OLHO DIR", "SORRIR", "GIRAR CABECA"]
desafio_atual = random.choice(testes)
liveness_aprovado = False
nome_identificado = ""

img_rosto = face_recognition.load_image_file(arquivo_rosto)
encoding_rosto = face_recognition.face_encodings(img_rosto)[0]

#MediaPipe
base_options = python.BaseOptions(model_asset_path=path_modelo)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

#Indices MediaPipe
L_EYE = {'top_bot': [159, 145], 'esq_dir': [33, 133]}
R_EYE = {'top_bot': [386, 374], 'esq_dir': [362, 263]}
MOUTH = {'top_bot': [13, 14], 'esq_dir': [78, 308]}
NOSE_TIP = 1
FACE_LEFT = 234
FACE_RIGHT = 454

def calcula_ratio(landmarks, pontos):
    v = distance.euclidean((landmarks[pontos['top_bot'][0]].x, landmarks[pontos['top_bot'][0]].y),(landmarks[pontos['top_bot'][1]].x, landmarks[pontos['top_bot'][1]].y))
    h = distance.euclidean((landmarks[pontos['esq_dir'][0]].x, landmarks[pontos['esq_dir'][0]].y),(landmarks[pontos['esq_dir'][1]].x, landmarks[pontos['esq_dir'][1]].y))
    return v/h

cap = cv2.VideoCapture(video_input)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h_f, w_f, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    result = detector.detect_for_video(mp_image, int(time.time() * 1000))

    if not result.face_landmarks:
        #Reset liveness
        liveness_aprovado = False
        desafio_atual = random.choice(testes)
        nome_identificado = ""
        status_msg = "Aguardando Rosto..."
        cor_ui = (255, 255, 255)
    else:
        landmarks = result.face_landmarks[0]
        status_msg = f"DESAFIO DE LIVENESS: {desafio_atual}"
        cor_ui = (0, 255, 255) #AMARELO
        #Calculo ratio
        ear_l = calcula_ratio(landmarks, L_EYE)
        ear_r = calcula_ratio(landmarks, R_EYE)
        mar = calcula_ratio(landmarks, MOUTH)
        
        #CALCULO GRIO YAW
        nose_x = landmarks[NOSE_TIP].x
        face_l_x = landmarks[FACE_LEFT].x
        face_r_x = landmarks[FACE_RIGHT].x
        rel_pos_nose = (nose_x - face_l_x) / (face_r_x - face_l_x)

        if not liveness_aprovado:
            if desafio_atual == "PISCAR OLHO ESQ" and ear_l < EAR_THRESHOLD and ear_r > EAR_THRESHOLD:
                liveness_aprovado = True
            elif desafio_atual == "PISCAR OLHO DIR" and ear_r < EAR_THRESHOLD and ear_l > EAR_THRESHOLD:
                liveness_aprovado = True
            elif desafio_atual == "SORRIR" and mar > MAR_THRESHOLD:
                liveness_aprovado = True
            elif desafio_atual == "GIRAR CABECA" and (rel_pos_nose < 0.35 or rel_pos_nose > 0.65):
                liveness_aprovado = True
        
        if liveness_aprovado:
            status_msg = "LIVENESS OK!"
            cor_ui = (0, 255, 0) # Verde
            xs = [int(l.x * w_f) for l in landmarks]
            ys = [int(l.y * h_f) for l in landmarks]
            top, right, bottom, left = min(ys), max(xs), max(ys), min(xs)
            
            loc = [(max(0, top-20), min(w_f, right+20), min(h_f, bottom+20), max(0, left-20))]
            encodings = face_recognition.face_encodings(rgb_frame, loc)
            
            if encodings:
                dist = face_recognition.face_distance([encoding_rosto], encodings[0])[0]
                nome_identificado = f"PEDRO ({dist:.2f})" if dist < 0.6 else "DESCONHECIDO"
                cv2.rectangle(frame, (left, top), (right, bottom), cor_ui, 2)
    #Exibe
    cv2.rectangle(frame, (0, 0), (w_f, 60), (0, 0, 0), -1)
    cv2.putText(frame, status_msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_ui, 2)
    if nome_identificado:
        cv2.putText(frame, nome_identificado, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Sistema Challenge-Response', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
