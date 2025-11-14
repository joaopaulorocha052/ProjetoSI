import face_recognition
import cv2
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
import dlib
from time import time

# Índices dos olhos
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(bocaS, bocaE) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(narizS, narizE) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

#Parametros
piscou = 0.2 #Detectar piscada
frames_consecutivos = 2 #Frames para detectar uma piscada
contador = 0
piscadas = 0
Liveness_Detectado = False

def calculate_ear(eye):
    #Calcula vertical
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    #Calcula horizontal
    C = distance.euclidean(eye[0], eye[3])
    #Calcula o EAR
    ear = (A + B) / (2.0 * C)
    return ear

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
except:
    print(f"Erro no predictor")
    exit()
try:
    rosto = face_recognition.load_image_file("pedro.jpg")
    rostos_conhecidos = face_recognition.face_encodings(rosto)
except:
    print("Nao encontrou rosto")
    exit()

nomes_rostos = ["Pedro"]

cap = cv2.VideoCapture(0)

#Variaveis
face_locations = []
face_encodings = []
face_names = []
processa = True
quadro_novo = 0
quadro_velho = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0) #usa modelo
    liveness_status = "Nao Liveness"
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        #Extrai coordenadas
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        boca = shape[bocaS:bocaE]
        nariz = shape[narizS:narizE]
        
        #Calcula EAR
        leftEAR = calculate_ear(leftEye)
        rightEAR = calculate_ear(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        rosto_inteiro = cv2.convexHull(shape)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        boca = cv2.convexHull(boca)
        nariz = cv2.convexHull(nariz)
        cv2.drawContours(frame, [rosto_inteiro], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [boca], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [nariz], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #Detectar piscada
        if ear < piscou:
            contador += 1
        else:
            if contador >= frames_consecutivos:
                piscadas += 1
                Liveness_Detectado = True
            contador = 0
        if Liveness_Detectado:
            liveness_status = f"Liveness Confirmado (piscadas: {piscadas})"
        else:
            liveness_status = "Esperando piscada"
    
    #exibe
    cv2.putText(frame, liveness_status, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    #entra no naive
    if Liveness_Detectado:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        if processa:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            if not face_locations:
                Liveness_Detectado = False
                piscadas = 0
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                #ve face conhecido
                matches = face_recognition.compare_faces(rostos_conhecidos, face_encoding)
                name = "Desconhecido"
                face_distances = face_recognition.face_distance(rostos_conhecidos, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = nomes_rostos[best_match_index]
                face_names.append(name)
        processa = not processa

    #Exibe e calc fps
    quadro_novo = time()
    fps = str(int(1/(quadro_novo-quadro_velho)))
    font = cv2.FONT_HERSHEY_DUPLEX
    quadro_velho = quadro_novo
    cv2.putText(frame, fps, (1860, 1070), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    cv2.imshow('Reconhecimento Facial (Vulnerável)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()