% ------------------------------------------------------------------------ 
%  Copyright (C)
%  Torr Vision Group (TVG)
%  University of Oxford - United Kingdom
% 
%  Anurag Arnab, Qizhu Li
%  February 2018
% ------------------------------------------------------------------------ 

function [pq, segmentation_quality, detection_quality] = compute_pq (overlaps, gt_counter, threshold)
% compute_pq: computes the panoptic quality metric.
% Arguements:
%   overlaps: The IoU of each prediction wrt to the ground truth
%   gt_counter: The number of ground truth segments
%   threshold: The prediction is considered correct if the IoU is greater than this
    
    correct_mask = (overlaps > threshold);
    tp = sum(correct_mask(:));
    fp = sum(~correct_mask(:));
    fn = gt_counter - tp;
    
    detection_quality = tp / (tp + 0.5*fp + 0.5*fn);
    
    segmentation_quality = 0;
    tp_ious = overlaps(correct_mask);
    if ~isempty(tp_ious)
        segmentation_quality = mean(tp_ious);
    end
    
    pq = segmentation_quality * detection_quality;
    
end