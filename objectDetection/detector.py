import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


from utils import label_map_util

from utils import visualization_utils as vis_util
import face_recognition
import cv2
import glob

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)

# Create arrays of known face encodings and their names
known_face_encodings = []

# Load a sample picture and learn how to recognize it.
path, dirs, files = next(os.walk("/home/jaspreet/Desktop/EasyBills/images"))
#print(os.path.exists("/home/el/myfile.txt"))

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
known_face_names = []

filenames = glob.glob("/home/jaspreet/Desktop/EasyBills/images" + "/*.jpg") #read all files in the path mentioned
for x in filenames:
  x = x.replace("/home/jaspreet/Desktop/EasyBills/images/", "")
  x = x.replace(".jpg", "")
  known_face_names.append(x)



# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


# # Model preparation 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model

if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
	print ('Downloading the model')
	opener = urllib.request.URLopener()
	opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
	tar_file = tarfile.open(MODEL_FILE)
	for file in tar_file.getmembers():
	  file_name = os.path.basename(file.name)
	  if 'frozen_inference_graph.pb' in file_name:
	    tar_file.extract(file, os.getcwd())
	print ('Download complete')
else:
	print ('Model already exists')

# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#intializing the web camera device

import cv2
cap = cv2.VideoCapture(0)

# Running the tensorflow session
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
   ret = True
   while (ret):
      ret,image_np = cap.read()
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      #print(scores.item(0))
      #print(classes.item(0))
      # dont make box for persons only for bottle
      #if not((scores.item(0))>.50 and (classes.item(0)==1)):
      #if(classes.item(0)==44):
      vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
      if(classes.item(0)==44):
        print("ID0's cart contains bottle weighing 70 grams")
      if(classes.item(0)==77):
        print("ID1's cart contains Samsung mobile weighing 570 grams")  
#      plt.figure(figsize=IMAGE_SIZE)
#      plt.imshow(image_np)
      #cv2.imshow('image',cv2.resize(image_np,(1280,960)))

    # Resize frame of video to 1/4 size for faster face recognition processing
      small_frame = cv2.resize(image_np, (0, 0), fx=0.25, fy=0.25)

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

              face_names.append(name)

      process_this_frame = not process_this_frame


      # Display the results
      for (top, right, bottom, left), name in zip(face_locations, face_names):
          # Scale back up face locations since the frame we detected in was scaled to 1/4 size
          top *= 4
          right *= 4
          bottom *= 4
          left *= 4

          # Draw a box around the face
          cv2.rectangle(image_np, (left, top), (right, bottom), (0, 0, 255), 2)

          # Draw a label with a name below the face
          cv2.rectangle(image_np, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
          font = cv2.FONT_HERSHEY_DUPLEX
          cv2.putText(image_np, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

      # Display the resulting image
      cv2.imshow('Video', image_np)

      # Hit 'q' on the keyboard to quit!
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  # Release handle to the webcam
  video_capture.release()
  cv2.destroyAllWindows()














