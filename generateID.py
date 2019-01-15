import numpy as np
import cv2


palm_cascade = cv2.CascadeClassifier(r'/home/jaspreet/Desktop/haar-face.xml')
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    cv2.imshow("frame",frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    palmy = palm_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in palmy:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        faces = frame[y:y+h,x:x+w]
	cv2.imwrite( "/home/jaspreet/Desktop//images/1", faces);
        #print(img)
        cv2.imshow('frame',faces)
    if cv2.waitKey(1) & 0xFF == ord('h'):
        break
   
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
