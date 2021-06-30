model: http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz

https://hub.tensorflow.google.cn/tensorflow/retinanet/resnet50_v1_fpn_640x640/1

https://docs.openvinotoolkit.org/latest/omz_models_model_ssd_resnet50_v1_fpn_coco.html

tf model conversion: 
--python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --reverse_input_channels --tensorflow_use_custom_operations_config/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config

export EXTENSION=/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/

python main.py -m model/frozen_inference_graph.xml -i resources/Pedestrian_Detect_2_1_1.mp4 -l $EXTENSION/libcpu_extension_avx2.so

python main.py -m model/frozen_inference_graph.xml -i resources/cropped5sec.mp4 -l $EXTENSION/libcpu_extension_avx2.so

### SSD Mobile Net
http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --reverse_input_channels --tensorflow_use_custom_operations_config/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config

python main.py -m model/frozen_inference_graph.xml -i resources/Pedestrian_Detect_2_1_1.mp4 -l $EXTENSION/libcpu_extension_avx2.so -pt 0.001


# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The process behind converting custom layers involves...

Some of the potential reasons for handling custom layers are...

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

Each of these use cases would be useful because...

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
