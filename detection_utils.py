import numpy as np
import cv2
import os
import datetime
import pdb

import matplotlib
matplotlib.use('agg')  # So that we can run this on the server
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms

import voc
CLASSES = voc.get_label_names()

def read_detection (detection_file):
    """Parses a saved detection file."""

    class_labels = []
    scores = []

    if not os.path.isfile(detection_file):
        raise AssertionError("Detection file " + detection_file + " does not exist")        

    f = open(detection_file, 'r')
    lines = f.readlines()

    num_detections = len(lines) // 3
    bboxes = np.zeros((num_detections, 4))

    for i in range(num_detections):
        class_label = int(lines[3*i])
        class_labels.append(class_label)

        score = float(lines[3*i+1])
        scores.append(score)
        bboxes[i, :] = [float(x) for x in lines[3*i+2].split()]

    return class_labels, scores, bboxes

def begin_file_write(filename):
    
    file_pointer = open(filename, 'w')
    return file_pointer
    
def write_detection(file_pointer, label, score, bbox):
    
    file_pointer.write(str(label) + "\n")
    file_pointer.write(str(score) + "\n")
    file_pointer.write(str(bbox[0]) + " " + str(bbox[1]) + " " + str(bbox[2]) + " " + str(bbox[3]) + "\n")
        
def close_file_write(file_pointer):
    
    file_pointer.write("0")
    file_pointer.close()

def vis_detections(im, class_name, dets, thresh=0.5, save_path='', box_colour = None,                                caption_bg_colour=None):
    """Draw detected bounding boxes."""

    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    if box_colour is None:
        box_colour = 'red'

    if caption_bg_colour is None:
        caption_bg_colour = 'blue'

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=box_colour, linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor=caption_bg_colour, alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    if not save_path:
        plt.draw()
    else:
        plt.savefig(save_path)
    

def detect_single(net, im, box_output_path, visualise_window=False, visualise_output_path=None, 
                  conf_thresh=0.7, nms_thresh=0.3, gpu=0):
    """Runs detector on an image

    Args:
        net: Caffe network to process the image
        im: a C x W x H image
        box_output_path: Path to write a plain-text file with saved detections.
        visualise_window: Boolean. Visualise the detection result in a separate window.
        visualise_output_path: Path to write an image with detections visualised.
            Visualised result not written if this is None.
        conf_thresh: Threshold to use for filtering detections
        nms_thresh: Threshold to use for non-maximal suppression.
        gpu: The GPU id to use.
    """

    im = im.astype(np.uint8)
    
    cfg.TEST.HAS_RPN = True
    cfg.GPU_ID = gpu if gpu >= 0 else -1

    scores, boxes = im_detect(net, im)
    
    box_output_file = begin_file_write(box_output_path)                        
    
    for cls_ind, cls in enumerate(CLASSES[1:]):

        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]

        keep = np.where(cls_scores >= conf_thresh)[0]
        cls_boxes = cls_boxes[keep, :]
        cls_scores = cls_scores[keep]

        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis]))
        dets = dets.astype(np.float32)
        keep = nms(dets, nms_thresh)
        dets = dets[keep, :]
        
        for i in range(dets.shape[0]):            
            write_detection(box_output_file, cls_ind, dets[i][4], dets[i][0:4])
        
        if (visualise_window or visualise_output_path):
            colour = voc.get_colour_map()
            idx = cls_ind
            colour = colour[3*idx:3*idx+3]    
            colour = [ float(x) / 256.0 for x in colour]

            vis_detections(im, cls, dets, conf_thresh, visualise_output_path, 
                           box_colour=colour, caption_bg_colour=colour)
    
    if visualise_window:
        plt.show()
    
    if visualise_window or visualise_output_path:
        plt.close('all')
        
    close_file_write(box_output_file)