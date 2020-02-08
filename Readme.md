# Pixelwise Instance Segmentation with a Dynamically Instantiated Network

This repository implements the panoptic/instance segmentation algorithm described in our CVPR 2017 paper, [Pixelwise Instance Segmentation with a Dynamically Instantiated Network](http://www.robots.ox.ac.uk/~aarnab/projects/cvpr_2017/Arnab_CVPR_2017.pdf).
Note that the term "panoptic segmentation" had not been coined at the time that this paper was originally written.

Subsequent work, in our ECCV 2018 paper, [Weakly and Semi-Supervised Panoptic Segmentation](https://github.com/qizhuli/Weakly-Supervised-Panoptic-Segmentation), showed how this model could be trained without any pixel-level supervision.

## Installation

Install our [Caffe fork](https://github.com/torrvision/caffe-tvg) with the Python bindings and ensure it is present in the ``$PYTHONPATH`` environment variable.

## Demo on arbitrary images

Download the pretrained models with `scripts/download_models.sh`. Then run `scripts/demo.sh`.

Note that this algorithm requires object detections as an additional input. This demo uses the public [R-FCN](https://github.com/YuwenXiong/py-R-FCN) model as the detector. 

## Results on Pascal VOC

This will reproduce the results of our fully-supervised model, pretrained on COCO, in [ECCV 2018 paper](http://www.robots.ox.ac.uk/~aarnab/projects/eccv_2018/Weakly_Sup_Panoptic_Seg_ECCV_2018.pdf) on the Pascal VOC dataset.

### Set-up
1. Download the Pascal VOC 2012 dataset, and place it (or a symlink to it) in `data`, such that, for example, `data/VOC2012/JPEGImages/2007_000033.jpg` is a valid path.
2. Download the pretrained models with `scripts/download_models.sh`.  
3. Download detection files with `scripts/download_voc_detections.sh`. If using your own detector, be careful to not train on the Pascal VOC validation set.

Run `scripts/experiment_voc.sh`

## Reference
If you find this code useful, please cite

```
 @inproceedings{arnab_cvpr_2017,
  author = {Anurag Arnab and Philip H. S. Torr},
  title = {Pixelwise Instance Segmentation with a Dynamically Instantiated Network},
  booktitle = {Computer Vision and Pattern Recognition (CVPR)},
  year = {2017}
}

 @inproceedings{li_eccv_2018,
  author = {Qizhu Li and Anurag Arnab and Philip H. S. Torr},
  title = {Weakly- and Semi-Supervised Panoptic Segmentation},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2018}
}
```

## Credits
The object detections used as an additional input to this algorithm are obtained using code from [Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn) and [R-FCN](https://github.com/YuwenXiong/py-R-FCN).

## Contact
For any queries, contact anurag.arnab@gmail.com. Pull requests are also welcome.