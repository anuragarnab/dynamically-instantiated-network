% ------------------------------------------------------------------------ 
%  Copyright (C)
%  Torr Vision Group (TVG)
%  University of Oxford - United Kingdom
% 
%  Anurag Arnab, Qizhu Li
%  February 2018
% ------------------------------------------------------------------------ 

function [results] = compute_metrics(prediction_root, opts)
%   compute_metrics: Compute evaluation metrics for instance- and panoptic-segmentation.
%   Args:
%       prediction_root: The directory with the predicted results
%       opts: Struct of evaluation options.
%   Returns:
%       results: A struct containing the mean average precision for instance segmentation,
%           the oracle mAP, which is the mAP that would be obtained if the predictions had
%           the optimal ordering, the panoptic quality for panoptic segmentation.
%           The panoptic quality is the product of two terms, the "segmentation quality"
%           and "detection quality" which are both returned.

    class_range = opts.class_range;
    iou_threshes = opts.iou_threshes;
    experiment_name = opts.experiment_name;

    mAPs = [];
    oracle_mAPs = [];
    panoptic_qualities = [];
    segmentation_qualities = [];
    detection_qualities = [];
    eval_data = [];
    
    for c = 1:numel(class_range)
        class_id = class_range(c);
        [all_overlaps, all_scores, gt_counter] = compute_overlaps(prediction_root, class_id, opts);
       
        for i = 1:numel(iou_threshes)
           thresh = iou_threshes(i);
           [mAP, oracle_mAP] = compute_mAP(all_overlaps, all_scores, gt_counter, thresh);
           mAPs(c, i) = mAP;
           oracle_mAPs(c, i) = oracle_mAP;
                      
           [pq, sq, dq] = compute_pq(all_overlaps, gt_counter, thresh);
           panoptic_qualities(c,i) = pq;
           segmentation_qualities(c,i) = sq;
           detection_qualities(c,i) = dq;
        end

        eval_data(c).overlaps = all_overlaps;
        eval_data(c).class_id = class_id;
        eval_data(c).scores = all_scores;
        eval_data(c).gt_counter = gt_counter;
        fprintf('Completed %d/%d classes \n', class_id, numel(class_range));
    end
    
    % Pack all the results into a struct
    results = [];
    results.iou_threshes = iou_threshes;
    results.mAPs = mAPs;
    results.oracle_mAPs = oracle_mAPs;
    results.eval_data = eval_data;
    results.panoptic_qualities = panoptic_qualities;
    results.segmentation_qualities = segmentation_qualities;
    results.detection_qualities = detection_qualities;
    
    fprintf('\n');
    % Print out the results
    if (opts.print_results)
        fprintf('\nIoU thresholds =\n\t%s\n\n', num2str(iou_threshes));
        fprintf('\nmean AP at above thresholds =\n\t%s\n\n', num2str(mean(mAPs), '%0.4f     ') );
        fprintf('\nOverall mean AP =\n\t%s\n\n', num2str(mean(mean(mAPs)), '%0.4f     ') );
        
        fprintf('\nmean Oracle AP at above thresholds =\n\t%s\n\n', num2str(mean(oracle_mAPs), '%0.4f     ') );
        fprintf('\nOverall mean Oracle AP =\n\t%s\n\n', num2str(mean(mean(oracle_mAPs)), '%0.4f     ') );

        fprintf('\nmean PQ at above thresholds =\n\t%s\n\n', num2str(mean(panoptic_qualities), '%0.4f     ') );
        fprintf('\nOverall mean PQ =\n\t%s\n\n', num2str(mean(mean(panoptic_qualities)), '%0.4f     ') );

        fprintf('\nmean SQ at above thresholds =\n\t%s\n\n', num2str(mean(segmentation_qualities), '%0.4f     ') );
        fprintf('\nOverall mean SQ =\n\t%s\n\n', num2str(mean(mean(segmentation_qualities)), '%0.4f     ') );
               
        fprintf('\nmean DQ at above thresholds =\n\t%s\n\n', num2str(mean(detection_qualities), '%0.4f     ') );
        fprintf('\nOverall mean DQ =\n\t%s\n\n', num2str(mean(mean(detection_qualities)), '%0.4f     ') );
    end
end