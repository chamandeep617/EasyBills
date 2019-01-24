import argparse
import imutils
import time
import cv2

from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream

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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
vs = VideoStream(src=0).start()
time.sleep(2.0)


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


# Running the tensorflow session
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
   ret = True
   while (ret):
     
      image_np = vs.read()
      if W is None or H is None:
        (H, W) = image_np.shape[:2]
        blob = cv2.dnn.blobFromImage(image_np, 1.0, (W, H),(104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        rects = []
      

      for i in range(0, detections.shape[2]):
        if detections[0, 0, i, 2] > args["confidence"]:
          box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
          rects.append(box.astype("int"))
          (startX, startY, endX, endY) = box.astype("int")
          cv2.rectangle(image_np, (startX, startY), (endX, endY),(0, 255, 0), 2)
      objects = ct.update(rects)
      
      for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        cv2.putText(image_np, text, (centroid[0] - 10, centroid[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(image_np, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        

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
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
#      plt.figure(figsize=IMAGE_SIZE)
#      plt.imshow(image_np)
      cv2.imshow("Frame", image_np)

      #cv2.imshow('image',cv2.resize(image_np,(1280,960)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          cap.release()
          break




