import numpy as np
import cv2
import imutils

palm_cascade = cv2.CascadeClassifier(r'D://MLs//ML_Haar_cascade_code//haarcascade_fullbody.xml')
#cap = cv2.VideoCapture("D://MLs//videoplayback.mp4")
while(True):
    ret, frame = cap.read()
    #frame = imutils.resize(frame, width=1000)
    #cv2.imshow("frame",frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    palmy = palm_cascade.detectMultiScale(gray, 1.1, 4)

    #if(len(palmy)!=0):
    for (x,y,w,h) in palmy:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255, 0, 0),2)
        cv2.putText(frame, "****************ID: 001****************", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('h'):
        break
   
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()