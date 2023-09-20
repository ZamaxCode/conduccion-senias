from ultralytics import YOLO
import cv2

model = YOLO('./best.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    predict = model.predict(frame, imgsz=640, conf=0.85)

    result = predict[0].plot()

    cv2.imshow("Conduccion", frame)

    if cv2.waitKey == 27:
        break

cap.release()
cap.destoryAllWindows()

