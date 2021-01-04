import numpy as np
import cv2
import os

name = input('Enter your name: ').lower().replace(' ','_')
print('Press s to capture your face. We need 20 frames of your face.')
print('Press q to cancel.')

register_path = './src/registered_faces/'
counter = 1

if not os.path.exists(register_path):
  os.mkdir(register_path)

new_register_path = os.path.join(register_path,name)

if os.path.exists(new_register_path):
  print('Name already exists.')
else:
  os.mkdir(new_register_path)
  face_cascade = cv2.CascadeClassifier('src/haar_cascades/haarcascade_frontalface_default.xml')
  cap = cv2.VideoCapture(0)

  while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
      cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
      face_roi_gray = gray[y:y+h,x:x+w]
    cv2.imshow('Face detection',frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):    
      filename = name+str(counter)+'.jpg'
      # save face roi in grayscale
      cv2.imwrite(os.path.join(new_register_path,filename),face_roi_gray)
      print('Picture',counter,'done.')
      counter+=1

    if counter>20:
      print('Done capturing.')
      break


  cap.release()
  cv2.destroyAllWindows()