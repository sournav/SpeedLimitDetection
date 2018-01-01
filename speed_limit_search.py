


# # Speed Limit Detection Demo


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image,ImageEnhance,ImageFilter
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
if tf.__version__ != '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')


sys.path.append("..")




from utils import label_map_util

from utils import visualization_utils as vis_util





# What model to use.
MODEL_NAME = 'speed_limit_graph'


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object_detection.pbtxt')

NUM_CLASSES = 1





detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')





label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)




def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)




# Add path to images directory below
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 2) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)





with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    i=1
    for image_path in TEST_IMAGE_PATHS:
      
      image = Image.open(image_path)
      img = cv2.imread(image_path)
      height, width, channels = img.shape
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      #finding which box is the bounding box
      y1=0
      y2=0
      x1=0
      x2=0
      boxy=np.squeeze(boxes)
      score=np.squeeze(scores)
      for i in range(boxy.shape[0]):
        if score[i]>=0.2:
            y1,x1,y2,x2=tuple(boxy[i].tolist())
      
      y1=int(round(y1*height))
      x1=int(round(x1*width))
      y2=int(round(y2*height))
      x2=int(round(x2*width))
      #cropping the image to the box
      im=Image.open(image_path)
      im=im.crop((x1,y1,x2,y2))
      #plt.figure(figsize=IMAGE_SIZE)
      #plt.imshow(im)
      enhancer = ImageEnhance.Brightness(im)
      factor=3.0
      im=enhancer.enhance(factor)
      
      #im=im.filter(ImageFilter.SMOOTH)
      
      
      im.save("pic1.png","png")
      im2=Image.open("pic1.png")
      text = pytesseract.image_to_string(im2)
      print(text)
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      
      #plt.figure(figsize=IMAGE_SIZE)
      #plt.imshow(image_np)
      cv2.imshow("detected",image_np)
      #plt.figure(figsize=IMAGE_SIZE)
      #plt.imshow(im)
      
      

