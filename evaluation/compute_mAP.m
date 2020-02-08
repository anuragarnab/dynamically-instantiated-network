% ------------------------------------------------------------------------ 
%  Copyright (C)
%  Torr Vision Group (TVG)
%  University of Oxford - United Kingdom
% 
%  Bernardino Romera Paredes, Anurag Arnab, Qizhu Li
%  November 2015
% ------------------------------------------------------------------------ 

function [mAP, mAP_oracle] = compute_mAP (overlaps, all_scores, gt_counter, threshold)
%   compute_mAP: computes the mean average precision
%   Args:
%       overlaps: The IoU of each prediction wrt to the ground truth
%       all_scores: The score of each prediciton
%       gt_counter: The number of ground truth segments
%       threshold: The prediction is considered correct if the IoU is greater than this
%   Returns:
%       mAP: The mean average precision
%       mAP_oracle: The best mAP that could be obtained by optimally ordering the
%           predictions.
    
    good_ones = (overlaps > threshold);    

    [josebi, order] = sort(all_scores, 'descend');
    good_ones = good_ones(order); 
    mAP = area_under_curve(good_ones) / gt_counter;
    
    oracle_good = zeros(size(good_ones));
    oracle_good(1:sum(good_ones)) = 1;
    mAP_oracle = area_under_curve(oracle_good) / gt_counter;

end


function [cum] = area_under_curve(good_ones)

    good_so_far = 0;
    cum = 0;
    for i = 1:length(good_ones)
        good_so_far = good_so_far + good_ones(i);
        cum = cum + good_ones(i)*(good_so_far/i);
    end

end