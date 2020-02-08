"""Panoptic/instance segmentation demo of the model described in:
A Arnab and P.H.S Torr, Pixelwise Instance Segmentation with a Dynamically Instantiated Network, CVPR 2017.

This model was then used as the fully-supervised baseline in:
Q Li*, A Arnab* and P.H.S Torr, Weakly and Semi-Supervised Panoptic Segmentation, ECCV 2018.

The provided model weights are taken from the Pascal VOC model from the ECCV 2018
paper trained on Pascal VOC and SBD. The model was pretrained using ImageNet and COCO.

This script can be used to reproduce the results of the fully-supervised baseline on
the Pascal VOC dataset in the ECCV 2018 paper.
Furthermore, the demo can also be run on arbitrary images.
"""

__author__ = 'Anurag Arnab'
__copyright__ = 'Copyright (c) 2019, Anurag Arnab'
__credits__ = ['Anurag Arnab', 'Qizhu Li', 'Philip Torr']
__email__ = 'anurag.arnab@gmail.com'
__license__ = 'MIT'

import argparse
import sys
import os
import shutil
import datetime

import caffe
import voc
import detection_utils

from PIL import Image as PILImage
import cv2
import numpy as np


def create_dir(dirname):
    try:
        os.makedirs(dirname)
    except OSError, e:
        if e.errno != os.errno.EEXIST:
            raise    


def parse_args():
    """Parse command line args."""
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_def', type = str)
    parser.add_argument('--model_weights', type = str)

    parser.add_argument('--image', type = str)
    parser.add_argument('--output_dir', type = str)
    parser.add_argument('--image_list', type = str, default = None)

    parser.add_argument('--model_det', type = str, default = 'models/rfcn.prototxt')
    parser.add_argument('--model_det_weights', type = str, 
                        default = 'models/rfcn.caffemodel')
    parser.add_argument('--det_threshold', type = float, default = 0.5)

    parser.add_argument('--cache_det_box', type = str, default = None)
    parser.add_argument('--force_overwrite', action = 'store_true', default = False)
    parser.add_argument('--nodetector', action='store_true', 
                        help='Pass the flag to disable the loading of the detector.'
                        'Raises error when a detection file does not exist.')
    
    parser.add_argument('--box_extension', type = str, default = '.bbox', 
                        help = 'The extension of bbox/detections files')
   
    parser.add_argument('--no-rescore_y', dest = 'rescore_y', action = 'store_false')
    parser.add_argument('--rescore_y', dest = 'rescore_y', action = 'store_true')
    parser.set_defaults(rescore_y = True)

    parser.add_argument('--gpu', type = int, default = -1)

    parser.add_argument('--pad_size', type = int, default = 500)
    parser.add_argument('--mean', type = str, default = '104.008,116.669,122.675')

    parser.add_argument('--blend_instance_output', action = 'store_true', default = False)
    parser.add_argument('--blend_alpha', type = float, default = 0.4)
    parser.add_argument('--vis_det', action = 'store_true', default = False)

    parser.add_argument('--iter_print', type = int, default = 50)

    args = parser.parse_args()
    sys.path.insert(0, 'fast_rcnn')

    args.mean = args.mean.split(',')
    args.mean = [float(x) for x in args.mean]
    args.mean = np.array(args.mean)
    
    args.palette = voc.get_colour_map()
    args.class_labels = voc.get_label_names()

    return args

def write_args(args):
    """Write command line arguements to the output directory."""
    
    filename = os.path.join(args.output_dir,
                            'cmdline' + str(datetime.datetime.now()))
    filename = filename.replace(' ','_') + '.txt'
    f_cmdline = open(filename, 'w')
    for i in range(len(sys.argv)):
        f_cmdline.write(sys.argv[i] + " ")


def log_timedmessage(message):
    """Print message with the time."""

    message = '{}\t{}'.format(str(datetime.datetime.now()), message.strip())
    print message
    sys.stdout.flush()


def save_image(pred, save_name, im_size, args):
    """Save segmentation result to a file with a colour palette.
    
    Args:
        pred: An image of type uint8.
        save_name: The path to save the result to.
        im_size: The original size of the image, as [h, w] The result will
            be resized to this size with nearest neighbour interpolation.
            In practice, this only applies if the image was resized down before
            processing.
        args: The command line arg object.
    """

    pred_image = PILImage.fromarray(pred)
    pred_image.putpalette(args.palette)

    # Resize the image to the correct size
    if ((pred_image.size[0] != im_size[1]) or pred_image.size[1] != im_size[0]):
        pred_image = pred_image.resize((im_size[1], im_size[0]),
                                        resample=PILImage.NEAREST)

    pred_image.save(save_name)


def rescore_y (y_variables, q, prediction=None, detected_classes=None):
    """Rescore the y_variables, ie, the score of each instance.

    Since the AP is a ranking metric, performance on it can be improved by
    rescoring each instance.
    This is done by averaging the semantic segmentation unaries within each
    instance.
    y_variables and q uses the same terminology as:
    "Pixelwise instance segmentation with a Dynamically Instantiated Network, 
    CVPR 2017."

    Args:
        y_variables: The detection scores for each instance.
        q: Of dimension C x H x W. The semantic segmentation unaries
            predicted by the network.
        prediction: The instance/panoptic segmentation of size H x W.
            Each pixel value is the instance id.
        detected_classes: The class label corresponding to each of the
            instance ids in "prediction"

    Returns:
        y_rescored: The rescored instance scores.
    """
    if (prediction is None):
        prediction = q.argmax(axis = 0).astype(np.uint8)

    y_variables = y_variables.ravel()
    y_rescored = np.zeros( y_variables.shape )

    for i, y_orig in enumerate(y_variables):
        index = i + 1
        q_index = index
        if (detected_classes is not None and len(detected_classes) > 0):
            q_index = detected_classes[i]
        
        mask = (prediction == index)

        y_rescored[i] = 0
        if (mask.sum() > 0):  # You can sum boolean arrays in numpy
            q_slice = np.squeeze( q[q_index,:,:] )
            y_rescored[i] = q_slice[mask].mean()

    return y_rescored


def postprocess_instance(instance_pred, y_variable, image, filename, args, 
                         im_size=None, semantic_seg_q=None):
    """Process the instance predictions for visualising and evaluation.

    The instance ids are processed to have consecutive numbers, and then
    saved to disk. This is necessary when some of the instances have no pixels
    assigned to them.
    This can occur when there are false-postive detections.
    A separate text file, which maps each instance id to a class label, and
    has the instance score, is also written for evaluation.
    Optionally, the original input image alpha-blended with the instance
    segmentation is written.

    Args:    
        instance_pred: An HxW matrix of integers. Each element indicates the
            instance id at a particular pixel.
        y_variable: The scores of each instance.
        image: The original image. Of size HxWx3.
        filename: The name of the original input image
        args: The command-line args object
        im_size: Size of the image as [H, W]
        semantic_seg_q: CxHxW matrix of the semantic segmentation unaries
            at each pixel.

    Returns:
        final_instance_pred: The final, HxW dimensional prediction.
    """
    
    dets_file_name = os.path.join(args.cache_det_box, filename + args.box_extension)
    class_labels, _, _ = detection_utils.read_detection( dets_file_name )
    instance_pred = instance_pred[0:image.shape[0], 0:image.shape[1]].astype(np.uint8)

    if args.rescore_y:
        y_variable = rescore_y(y_variable, semantic_seg_q, instance_pred, class_labels)

    # Postprocess to have consecutive numbers
    final_instance_pred = np.zeros(instance_pred.shape)
    uniques = np.unique(instance_pred)

    for i in range( len(uniques) ):
        indices = ( instance_pred == uniques[i] )
        final_instance_pred[indices] = i

    final_instance_pred = final_instance_pred[0:image.shape[0], 0:image.shape[1]].astype(np.uint8)    
    final_instance_im = PILImage.fromarray(final_instance_pred)
    final_instance_im.putpalette(args.palette)

    if (im_size != None):
        final_instance_im = final_instance_im.resize( (im_size[1], im_size[0]), resample = PILImage.NEAREST)

    final_instance_im.save( os.path.join( args.output_dir, filename + '.png') )

    # Write the scores to file
    y_variable = y_variable.ravel()

    txt_filename = os.path.join( args.output_dir, filename + ".txt")
    f = open(txt_filename, 'w')

    for i in range( len(uniques) ):
        index = uniques[i]
        if (index == 0):
            continue
        y = y_variable[index-1]
        class_label = class_labels[index-1]

        f.write( str(class_label) + " " + str(y) + "\n")
    f.close()   

    if (args.blend_instance_output):
        palette = np.array(args.palette).reshape( (len(args.palette)/3,3) ).astype(np.uint8)
        result_image = palette[final_instance_pred.astype(np.uint8).ravel()].reshape(image.shape)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        blend_image = image * args.blend_alpha + result_image * (1 - args.blend_alpha)

        cv2.imwrite(os.path.join(args.output_dir, filename + "_inst_blend.png"), blend_image)    

    return final_instance_pred


def preprocess_image(name, args, pad_width=None, pad_height=None, scale=None, 
                     pad_value=cv2.BORDER_REFLECT_101):
    """Pre-process the input image to feed into the network."""

    image_orig = cv2.imread(name, 1)
    
    if image_orig is None:
        raise AssertionError(name + " was not found.")

    image_orig = image_orig.astype(np.float32)
    image = image_orig - args.mean
    im_height, im_width = image.shape[0], image.shape[1]

    # Make image smaller, if it is bigger than pad_width x pad_height
    im_height = image.shape[0]
    im_width = image.shape[1]
    
    if (im_height > pad_height or im_width > pad_width):
        if (im_height > im_width):
            ratio = float( max(pad_width, pad_height) / float(im_height) )
        else:
            ratio = float( max(pad_width, pad_height) / float(im_width) )
        
        image = cv2.resize(
            image,
            dsize=(int(im_width * ratio), int(im_height * ratio)))
        image_orig = cv2.resize(
            image_orig, 
            dsize=(int(im_width * ratio), int(im_height * ratio)))

    if scale:
        if scale > 1:
            image = cv2.resize(image, dsize=(0,0), fx=scale, fy=scale,       
                               interpolation=cv2.INTER_CUBIC)
        if scale < 1:
            image = cv2.resize(image, dsize=(0,0), fx=scale, fy=scale, 
                               interpolation=cv2.INTER_AREA)

    if (pad_width is not None) and (pad_height is not None):
        image = cv2.copyMakeBorder(
            image, 0, pad_height - image.shape[0],
            0, pad_width - image.shape[1], pad_value)

    image = image.transpose([2, 0, 1]) # To make it C x H x W
    return image, image_orig, [im_height, im_width]


def process(net_segmentor, net_detector, image_filename, args):
    """Process a single image.
    
    Args:
        net_segmentor: The segmentation network
        net_detector: Detection network. The segmentor uses object detections
            as an additional input. Cached detections, if available, are read
            from disk. If unavailable, the detector is run. In practice, this
            is only used for processing arbitrary images, as for standard
            datasets, the detections are precomputed.
        image_filename: The image to process.
        args: Command-line args object.
    """
    
    fname = os.path.basename(image_filename)
    fname = fname.split(".")[0]

    if (os.path.isfile(os.path.join(args.output_dir, fname + ".png")) and not 
        args.force_overwrite):
        return

    image, image_orig, im_size = preprocess_image(
        image_filename, args, pad_width=args.pad_size, pad_height=args.pad_size)

    det_box_path = os.path.join(args.cache_det_box, fname + args.box_extension)
    if not os.path.exists(det_box_path):
        if net_detector is None:
            message = 'Detection file not present and detector not loaded'
            raise AssertionError(message)
        detection_utils.detect_single(net_detector, image_orig, det_box_path,
                                      conf_thresh=args.det_threshold, gpu=args.gpu)

    class_labels, scores, bboxes = detection_utils.read_detection(det_box_path)

    # The actual segmentation
    net_segmentor.blobs['data'].data[0,:,:,:] = image
    num_det = len(class_labels)
    detections = np.zeros((1, num_det, 1, 6))
    detections[0, :, 0, 0] = class_labels
    detections[0, :, 0, 1:5] = bboxes
    detections[0, :, 0, 5] = scores
    if net_segmentor.blobs['detections'].shape != (1, num_det, 1, 6):
        net_segmentor.blobs['detections'].reshape(1, num_det, 1, 6)
        net_segmentor.reshape()
    net_segmentor.blobs['detections'].data[...] = detections.astype(np.float32)

    if num_det == 0:
        net_segmentor.forward(end='Softmax_norm')
    else:
        net_segmentor.forward()

    pred_instance_seg = net_segmentor.blobs['instance_prediction'].data[0]
    pred_instance_seg = pred_instance_seg.argmax(axis = 0).astype(np.uint8)
    y_variables = net_segmentor.blobs['instance_y_variables'].data[...]

    semantic_seg_q = None
    if args.rescore_y:
        semantic_seg_q = net_segmentor.blobs['prob'].data[0]
        semantic_seg_q = semantic_seg_q[:, 0:image_orig.shape[0], 0:image_orig.shape[1]]

    postprocess_instance(pred_instance_seg, y_variables, image_orig, fname,
                         args, im_size, semantic_seg_q)
    

def main():
    args = parse_args()
    create_dir(args.output_dir)

    if args.cache_det_box:
        create_dir(args.cache_det_box)

    if args.gpu >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu)

    net_segmentor = caffe.Net(args.model_def, args.model_weights, caffe.TEST)
    if args.nodetector:
        net_detector = None
    else:
        net_detector = caffe.Net(args.model_det, args.model_det_weights, caffe.TEST)

    if args.image_list is None:
        process(net_segmentor, net_detector, args.image, args)
    else:
        write_args(args)
        
        image_names = open(args.image_list, 'r').readlines()
        num_im = len(image_names)
        count_im = 0

        for line in image_names:
            line = line.strip()
            args.image = line
            process(net_segmentor, net_detector, args.image, args)
            
            count_im += 1
            if (count_im % args.iter_print == 1):
                log_timedmessage('Processed {}/{} images.'.format(count_im, num_im))
    
    print 'Saved results to', args.output_dir


if __name__ == '__main__':
    main()
