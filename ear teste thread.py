import face_recognition
import cv2
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
import dlib
from time import time
from threading import Thread


class WebcamStream:
    def __init__(self, src=0):
        # Inicializa a câmera
        # No Mac, cv2.CAP_AVFOUNDATION pode ajudar a ser mais explícito, 
        # mas o padrão (0) geralmente funciona bem.
        self.stream = cv2.VideoCapture(src)
        
        # Definir resolução (opcional, mas recomendado no Mac para performance)
        # self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        (self.grabbed, self.frame) = self.stream.read()

        # Variável para controlar o loop da thread
        self.stopped = False

    def start(self):
        # Inicia a thread que lê os frames
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Loop infinito dentro da thread
        while True:
            # Se o método stop foi chamado, encerra a thread
            if self.stopped:
                self.stream.release()
                return

            # Caso contrário, lê o próximo frame do buffer
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Retorna o frame mais recente
        return self.grabbed,self.frame

    def stop(self):
        # Indica que a thread deve parar
        self.stopped = True

# Índices dos olhos
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

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

# --- Configuração Inicial (Linha de Base)
try:
    rosto = face_recognition.load_image_file("pedro.jpg")
    rostos_conhecidos = face_recognition.face_encodings(rosto)
except:
    print("Nao encontrou rosto")
    exit()

nomes_rostos = [
    "Pedro"
]


cap = WebcamStream(src=0).start()

# Inicializa variáveis para o processamento
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
    # Deteta faces no quadro (dlib)
    rects = detector(gray, 0)
    
    liveness_status = "Liveness Nao Verificado"
    
    # Itera sobre as faces detetadas
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extrai as coordenadas dos olhos
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        # Calcula o EAR para ambos os olhos
        leftEAR = calculate_ear(leftEye)
        rightEAR = calculate_ear(rightEye)


        # Calcula a média do EAR
        ear = (leftEAR + rightEAR) / 2.0

        # Desenha o contorno dos olhos (para debug)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Lógica de deteção de piscadela 
        if ear < piscou:
            contador += 1
        else:
            # Se os olhos estiveram fechados o suficiente...
            if contador >= frames_consecutivos:
                piscadas += 1
                Liveness_Detectado = True # Liveness verificado!
            # Resetar o contador
            contador = 0

        if Liveness_Detectado:
            liveness_status = f"Liveness Confirmado (Piscadelas: {piscadas})"
        else:
            liveness_status = "A piscar os olhos..."
    
    # Exibe o status do Liveness no quadro
    cv2.putText(frame, liveness_status, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # --- Portão Lógico ---
    # Só executa o reconhecimento (Sec 2.4) se o liveness for confirmado
    if Liveness_Detectado:
    # Captura um único quadro de vídeo

    # Redimensiona o quadro para 1/4 do tamanho para processamento mais rápido
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Converte a imagem de BGR (OpenCV) para RGB (face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Processa apenas um a cada dois quadros para poupar tempo
        if processa:
            # Encontra todas as faces e seus encodings no quadro atual
            face_locations = face_recognition.face_locations(rgb_small_frame)
            if not face_locations:
                Liveness_Detectado = False
                piscadas = 0
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # Vê se a face corresponde a alguma(s) face(s) conhecida(s)
                matches = face_recognition.compare_faces(rostos_conhecidos, face_encoding)
                name = "Desconhecido"

                # Usa a face conhecida com a menor distância para a nova face
                face_distances = face_recognition.face_distance(rostos_conhecidos, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = nomes_rostos[best_match_index]

                face_names.append(name)

        processa = not processa

    # --- Exibição dos Resultados ---

    # Desenha os resultados no quadro original (não redimensionado)
    quadro_novo = time()
    fps = str(int(1/(quadro_novo-quadro_velho)))
    font = cv2.FONT_HERSHEY_DUPLEX
    quadro_velho = quadro_novo
    cv2.putText(frame, fps, (1860, 1070), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Re-escala as localizações da face (pois detetamos num quadro 1/4)
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Desenha uma caixa ao redor da face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Desenha um rótulo com o nome abaixo da face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Exibe a imagem resultante
    cv2.imshow('Reconhecimento Facial (Vulnerável)', frame)
    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberta a webcam e fecha as janelas
cap.release()
cv2.destroyAllWindows()