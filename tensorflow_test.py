import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
import pathlib

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"
  print(model_dir)
  model = tf.saved_model.load(str(model_dir))
  model = model.signatures['serving_default']

  return model

  # List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/home/thisara/models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def run_inference_for_single_image(model, image, excecution_times):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  start = time.time()
  output_dict = model(input_tensor)
  end = time.time()
  

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

  count = [x for x in output_dict['detection_classes'] if x == 1]

  if(len(count) > 0):
    excecution_times.append((end - start) * 1000)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(output_dict['detection_masks'], output_dict['detection_boxes'],          image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict


model_name = 'ssd_mobilenet_v2_coco_2018_03_29'
detection_model = load_model(model_name)

input = '/home/thisara/Documents/Pedestrian_Detect_2_1_1.mp4'
vid_capture = cv2.VideoCapture(input)
vid_capture.open(input)

width = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vid_capture.get(cv2.CAP_PROP_FPS))
fcount = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))

print('{} meta data: width:{}, height:{}, fps:{}, fcount:{}'.format('Vedeo', width, height, fps, fcount))  

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter('/home/thisara/Documents/out.mp4', fourcc, fps, (width,height))

excecution_times = []

while(vid_capture.isOpened()):
    ret_val, frame = vid_capture.read()
    if(ret_val):
        output_dict = run_inference_for_single_image(detection_model, frame, excecution_times)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)

        out.write(frame) # Write frame for debuggin    
    else:
        break

out.release()

print('total count: {}, mean: {}'.format(np.mean(np.array(excecution_times)), len(excecution_times)))
