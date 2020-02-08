% ------------------------------------------------------------------------ 
%  Copyright (C)
%  Torr Vision Group (TVG)
%  University of Oxford - United Kingdom
% 
%  Anurag Arnab, Qizhu Li
%  February 2018
% ------------------------------------------------------------------------ 

function [opts] = get_mAP_options(dataset)
    %%Evaluation options for several segmentation datasets

    opts = [];
    opts.experiment_name = '';
    opts.debug = 0;
    opts.ignore_enabled = 1;
    opts.convert_to_indexed_png = 0;
    opts.ignore_label = 255;
    opts.verbose = 0;
    opts.iou_threshes = [0.1:0.1:0.9];
    opts.print_results = 1;
    opts.txt_results = '';
    opts.iter_print = 100;
    opts.not_eval_label = -1;
    opts.add_background_det = false;
        
    opts.pred_suffix = '.png';
    opts.anno_suffix = '.png';
    opts.score_suffix = '.txt';
            
    if (strcmp(dataset, 'voc2012'))
    
         opts.image_set = 'original_val';       
         opts.path_root = '../data/VOC2012/';
         opts.list_file = fullfile(opts.path_root, ['ImageSets/Segmentation/' opts.image_set '.txt']); 
         
         opts.annotation_instance_root = fullfile(opts.path_root, 'SegmentationObject');
         opts.annotation_semantic_root = fullfile(opts.path_root, 'SegmentationClass');
         opts.class_range = [1:1:20];
         
         opts.not_eval_label = 0;
    
    elseif (strcmp(dataset, 'sbd'))
    
         opts.image_set = 'sds/val';
         opts.path_root = 'SBD_Hariharan/VOC2011_berkeley/VOC2011/';
         opts.list_file = fullfile(opts.path_root, ['ImageSets/Segmentation/' opts.image_set '.txt']);
         
         opts.annotation_instance_root = fullfile(opts.path_root, 'SegmentationObject');
         opts.annotation_semantic_root = fullfile(opts.path_root, 'SegmentationClass');
         opts.class_range = [1:1:20];
         
         opts.not_eval_label = 0;

    elseif (strcmp(dataset, 'minisbd'))
    
         opts.image_set = 'sbdsplit_mini_val';
         opts.path_root = 'VOC2012/';
         opts.list_file = fullfile(opts.path_root, ['ImageSets/Segmentation/' opts.image_set '.txt']);
         
         opts.annotation_instance_root = fullfile(opts.path_root, 'SegmentationObject');
         opts.annotation_semantic_root = fullfile(opts.path_root, 'SegmentationClass');
         opts.class_range = [1:1:20];
         
         opts.not_eval_label = 0;
         
    elseif ( strcmp(dataset, 'pp_sbd') )
    
        opts.image_set = 'val_id_sbd';
        opts.path_root = 'pascal_person_part';
        opts.list_file = fullfile(opts.path_root, ['pascal_person_part_trainval_list/' opts.image_set '.txt']);
        
        opts.annotation_instance_root = 'SBD_Hariharan/VOC2011_berkeley/VOC2011/SegmentationObject';
        opts.annotation_semantic_root = 'SBD_Hariharan/VOC2011_berkeley/VOC2011/SegmentationClass';
        opts.class_range = [15];
        
        opts.not_eval_label = 0;
    
    elseif ( strcmp(dataset, 'sbd_person_val') )
        
        opts.image_set = 'sds/sbd_val_with_people';
        opts.path_root = 'SBD_Hariharan/VOC2011_berkeley/VOC2011/';
        opts.list_file = fullfile(opts.path_root, ['ImageSets/Segmentation/' opts.image_set '.txt']);

        opts.annotation_instance_root = fullfile(opts.path_root, 'SegmentationObject');
        opts.annotation_semantic_root = fullfile(opts.path_root, 'SegmentationClass');
        opts.class_range = [15];
        
        opts.not_eval_label = 0;

    if (strcmp(dataset, 'pascal_person'))

        opts.image_set = 'val_id';
        opts.path_root = 'pascal_person_part';
        opts.list_file = fullfile(opts.path_root, ['pascal_person_part_trainval_list/' opts.image_set '.txt']);
        
        opts.annotation_instance_root = 'pascal_person_part/SegmentationPartInstance/';
        opts.annotation_semantic_root = 'pascal_person_part/SegmentationPart/';
        opts.class_range = [1:1:6];

    elseif ( strcmp(dataset, 'voc2012_person_val') )
        
        opts.image_set = 'original_val';
        opts.path_root = 'VOC2012/';
        opts.list_file = fullfile(opts.path_root, ['ImageSets/Segmentation/' opts.image_set '.txt']);

        opts.annotation_instance_root = fullfile(opts.path_root, 'SegmentationObject');
        opts.annotation_semantic_root = fullfile(opts.path_root, 'SegmentationClass');
        opts.class_range = [15];
        
        opts.not_eval_label = 0;
            
        
    elseif (strcmp(dataset, 'cityscapes_panoptic'))
    
        opts.image_set = 'val_id';
        opts.path_root = 'Cityscapes';
        opts.list_file = fullfile(opts.path_root, 'lists', [opts.image_set '.txt']);
        opts.pred_suffix = '.png';
        opts.score_suffix = '.txt';
        
        opts.annotation_instance_root = fullfile(opts.path_root, 'gtFine_panoptic/val');
        opts.annotation_semantic_root = fullfile(opts.path_root, 'gtFine_semantic/val');
        opts.class_range = [0:18];
        
        opts.iou_threshes = [0.1:0.1:0.5, 0.55:0.05:0.95];
        
        opts.add_background_det = true;
        opts.background_det_score = 1;
        opts.background_class = 0;
        
        opts.iter_print = 50;
        
    elseif (strcmp(dataset, 'cityscapes_instance'))
    
        opts.image_set = 'val_id';
        opts.path_root = 'Cityscapes';
        opts.list_file = fullfile(opts.path_root, 'lists', [opts.image_set '.txt']);
        opts.pred_suffix = '.png';
        opts.score_suffix = '.txt';
        
        opts.annotation_instance_root = fullfile(opts.path_root, 'gtFine_panoptic/val');
        opts.annotation_semantic_root = fullfile(opts.path_root, 'gtFine_semantic/val');
        opts.class_range = [1:8];
        
        opts.iou_threshes = [0.1:0.1:0.5, 0.55:0.05:0.95];
        
        opts.iter_print = 50;
         
    else
        error('Unknown dataset')
    end

end
