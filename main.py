"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    # Note - CPU extensions are moved to plugin since OpenVINO release 2020.1. 
    # The extensions are loaded automatically while     
    # loading the CPU plugin, hence 'add_extension' need not be used.

 
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                             default=None,
                             help="MKLDNN (CPU)-targeted custom layers."
                                  "Absolute path to a shared library with the"
                                  "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = None

    return client

def pre_process(image, height, width):
    processed_image = cv2.resize(image, (width, height))
    processed_image = np.transpose(processed_image, (2, 0, 1))
    processed_image = np.expand_dims(processed_image, axis = 0)
    return processed_image

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)

    ### TODO: Handle the input stream ###
    vid_capture = cv2.VideoCapture(args.input)
    vid_capture.open(args.input)
    
    width = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid_capture.get(cv2.CAP_PROP_FPS))
    fcount = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print('Video frame meta data: width:{}, height:{}, fps:{}, fcount:{}'.format(width, height, fps, fcount))
    
    input_shape = infer_network.get_input_shape()
    out = cv2.VideoWriter('out.mp4', 0x00000021, fps, (width,height))
    
    if( not vid_capture.isOpened()):
        print("Error opening the video input")
        return
    
    ### TODO: Loop until stream is over ###
    while(vid_capture.isOpened()):
        ### TODO: Read from the video capture ###
        ret_val, frame = vid_capture.read()
        if(ret_val):
            
            ### TODO: Pre-process the image as needed ###
            pre_processed = pre_process(frame, input_shape[2], input_shape[3])
            
            ### TODO: Start asynchronous inference for specified request ###
            infer_network.exec_net(pre_processed)
            ### TODO: Wait for the result ###
            infer_network.wait()

            ### TODO: Get the results of the inference request ###
            results = infer_network.get_output()
            for index, conf  in enumerate(results[0,0,:, 2]):
                if(conf > args.prob_threshold): 
                    x_min = int(results[0, 0, index, 3] * frame.shape[1])
                    y_min = int(results[0, 0, index, 4] * frame.shape[0])
                    x_max = int(results[0, 0, index, 5] * frame.shape[1])
                    y_max = int(results[0, 0, index, 6] * frame.shape[0])
                    
                    if(results[0, 0, index, 1] == 1):
                        cv2.rectangle(frame, (x_min,y_min), (x_max,y_max), (255,0,0),  1) 
                        
            
            out.write(frame) # Write frame irrespective of result
                    

                    
                    
            ### TODO: Extract any desired stats from the results ###

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

        ### TODO: Send the frame to the FFMPEG server ###

        ### TODO: Write an output image if `single_image_mode` ###
        else:
            break
    
    out.release()
    print('Inference total latency in seconds: {}'.format(int(infer_network.get_latency()/1000)))



def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
