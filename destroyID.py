import numpy as np
import face_recognition
import cv2
import os
import glob
import webbrowser
url = './payment.html'


# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Create arrays of known face encodings and their names
known_face_encodings = []

def release():
    video_capture.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Load a sample picture and learn how to recognize it.
path, dirs, files = next(os.walk("/home/jaspreet/Desktop/EasyBills/images"))
img_count = len(files)
for x in range(10):
  print(x)
  if(os.path.exists("/home/jaspreet/Desktop/EasyBills/images/"+str(x)+".jpg")==True):
    (globals()["image" + str(x)]) = face_recognition.load_image_file("/home/jaspreet/Desktop/EasyBills/images/"+str(x)+".jpg")
    (globals()["face_encoding" + str(x)]) = face_recognition.face_encodings((globals()["image" + str(x)]))[0]

#print(face_encoding1)
# Load a second sample picture and learn how to recognize it.
#biden_image = face_recognition.load_image_file("/home/jaspreet/Pictures/Webcam/cs.jpg")
#biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

for x in range(10):
  if(os.path.exists("/home/jaspreet/Desktop/EasyBills/images/"+str(x)+".jpg")==True):
     known_face_encodings.append(globals()[("face_encoding")+str(x)])

#known_face_names = ["A","B" ]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

known_face_names = []

filenames = glob.glob("/home/jaspreet/Desktop/EasyBills/images" + "/*.jpg") #read all files in the path mentioned
for x in filenames:
  x = x.replace("/home/jaspreet/Desktop/EasyBills/images/", "")
  x = x.replace(".jpg", "")
  known_face_names.append(x)


while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
		print(name)
                os.remove('/home/jaspreet/Desktop/EasyBills/images/'+name+'.jpg')
                webbrowser.open(url, new=2)  # open in new tab
		print ("********* PAYMENT PORTAL STARTED ********* \n  ID destroyed :"+ str(name))
	        release()

            face_names.append(name)
            if cv2.waitKey(1) & 0xFF == ord('h'):
               break
    break

    process_this_frame = not process_this_frame

   

   
