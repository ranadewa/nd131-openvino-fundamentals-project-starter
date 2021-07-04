#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.net = None
        self.plugin = None
        self.execNetwork = None
        self.input_key = None
        self.output_key = None
        self.request_handle = None
        self.latency = 0
        ### TODO: Initialize any class variables desired ###

    def load_model(self, model_path, device, cpu_extension):
        ### TODO: Load the model ###
        bin_path = model_path.split('.')[0] + '.bin'

        self.net = IENetwork(model=model_path, weights=bin_path)
        self.input_key = next(iter(self.net.inputs.keys()))
        self.output_key = next(iter(self.net.outputs.keys()))
        log.info('Model meta data: ')
        log.info('\t input shape: {}'.format(self.net.inputs[self.input_key].shape))
        log.info('\t output shape: {}'.format(self.net.outputs[self.output_key].shape))
        self.plugin = IECore()
        self.plugin.add_extension(cpu_extension, device)
        
        ### TODO: Check for supported layers ###
        layerMap = self.plugin.query_network(self.net, device_name =device)
        
        for layer in layerMap.keys():
            if (layer not in self.net.layers.keys()):
                log.error("Unsupported layer found. Exiting")
                return
        
        self.execNetwork = self.plugin.load_network(self.net, device)

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.net.inputs[self.input_key].shape

    def exec_net(self, image):
        ### TODO: Start an asynchronous request ###
        self.request_handle = self.execNetwork.start_async(0, inputs = {
            self.input_key : image
        })

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return self.request_handle.wait()

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        self.latency += self.request_handle.latency
        return self.request_handle.outputs[self.output_key]
    
    def get_latency(self):
        return self.request_handle.latency
    
    def get_total_latency(self):
        return self.latency
