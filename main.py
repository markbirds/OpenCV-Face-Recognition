import numpy as np
import cv2
import json

# loading features to be used for face detection
face_cascade = cv2.CascadeClassifier('src/haar_cascades/haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("./src/recognizer/face_trained.yml")

with open('./src/recognizer/registered_faces.json') as f:
    registered_faces = json.load(f)['registered_faces']

# capture live feed from webcam
cap = cv2.VideoCapture(0)

while(True):
  # read frames and covert to grayscale
  ret, frame = cap.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # get coordinates of faces
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  for (x,y,w,h) in faces:
    # draw rectangle around face roi
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)    
    face_roi_gray = gray[y:y+h, x:x+w]

    id_, conf = face_recognizer.predict(face_roi_gray)
    font = cv2.FONT_HERSHEY_SIMPLEX
    name = registered_faces[id_]
    color = (255, 255, 255)
    stroke = 2
    cv2.putText(frame, name, (x,y-20), font, 1, color, stroke, cv2.LINE_AA)

  # display resulting frame
  cv2.imshow('Face detection',frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break


cap.release()
cv2.destroyAllWindows()