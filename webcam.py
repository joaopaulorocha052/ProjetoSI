import cv2
from time import time
quadro_novo = 0
quadro_velho = 0
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    quadro_novo = time()
    fps = 1/(quadro_novo-quadro_velho)
    quadro_velho = quadro_novo
    fps = str(int(fps))
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()