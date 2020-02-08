#!/bin/bash
cd fast_rcnn && make && cd -

IMAGE=data/demo/2007_001311.jpg
GPU=0

python demo.py \
--model_def models/din.prototxt \
--model_weights models/din_coco_voc.caffemodel \
--image ${IMAGE} --pad_size 521 \
--gpu ${GPU} \
--output_dir output/demo/ \
--cache_det_box data/demo/ \
--blend_instance_output
