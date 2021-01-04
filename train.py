import numpy as np
import cv2
import os
import json

registered_face_dir = './src/registered_faces'
registered_faces = [name for name in os.listdir(registered_face_dir)]

X_train = []
y_train = []

for index,name in enumerate(registered_faces):
  current_face_dir = os.path.join(registered_face_dir,name)
  for img_name in os.listdir(current_face_dir):
    img_path = os.path.join(current_face_dir,img_name)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    X_train.append(gray)
    y_train.append(index)


registered_faces_tosave = {
    'registered_faces': registered_faces
}

with open("src/recognizer/registered_faces.json","w") as f:
  f.write(json.dumps(registered_faces_tosave))


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(np.array(X_train,dtype=object),np.array(y_train))
face_recognizer.save("src/recognizer/face_trained.yml")