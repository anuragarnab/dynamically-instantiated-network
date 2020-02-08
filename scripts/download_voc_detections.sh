#!/bin/bash

# Checksum of the zip file should be 5eedfe8337b0e1cd6c099be896ef14af

wget -N www.robots.ox.ac.uk/~aarnab/projects/cvpr_2017/detections/boxes_voc.zip -P data/

mkdir -p data/detections
unzip data/boxes_voc.zip -d data/detections/
rm data/boxes_voc.zip