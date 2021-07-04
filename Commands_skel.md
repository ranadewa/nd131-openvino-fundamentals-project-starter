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

python main.py -m model/frozen_inference_graph.xml -i resources/Pedestrian_Detect_2_1_1.mp4 -l $EXTENSION/libcpu_extension_avx2.so -pt 0.05 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 10 -i - http://0.0.0.0:3004/fac.ffm
