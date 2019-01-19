import numpy as np
import cv2
import os

face_cascade = cv2.CascadeClassifier(r'/home/jaspreet/Desktop/EasyBills/HaarCascade Algorithm/haar-face.xml')
cap = cv2.VideoCapture(0)
   
def release():
    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

while(True):
    ret, frame = cap.read()
    cv2.imshow("frame",frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    facy = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in facy:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        faces = frame[y:y+h,x:x+w]
	path, dirs, files = next(os.walk("/home/jaspreet/Desktop/EasyBills/images"))
	new_img_ID = len(files)	
	cv2.imwrite( "/home/jaspreet/Desktop/EasyBills/images/"+str(new_img_ID)+".jpg", faces);
        #print(img)
        cv2.imshow('ID',faces)
	print ("ID generated :"+ str(new_img_ID))
	release()
    if cv2.waitKey(1) & 0xFF == ord('h'):
        break

