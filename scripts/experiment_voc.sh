#!/bin/bash
cd fast_rcnn && make && cd -

OUTPUT_DIR=output/voc2012_val/din
GPU=0

python demo.py --model_def models/din.prototxt \
--model_weights models/din_coco_voc.caffemodel \
--nodetector --cache_det_box data/detections/ \
--pad_size 521 \
--image_list data/voc2012_val.txt \
--gpu ${GPU} \
--output_dir ${OUTPUT_DIR} 

# Evaluation
MATLAB=matlab
COMMAND="cd evaluation; run_compute_metrics('../${OUTPUT_DIR}'); quit"
${MATLAB} -nosplash -nodesktop -r "${COMMAND}"