import face_recognition
import cv2
import numpy as np
from time import time

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
    
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    if processa:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            #Ve face conhecido
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
    fps = 1/(quadro_novo-quadro_velho)
    quadro_velho = quadro_novo
    fps = str(int(fps))
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    cv2.imshow('Reconhecimento Facial (Vulnerável)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()