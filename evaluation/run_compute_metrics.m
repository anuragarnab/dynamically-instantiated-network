function [results] = run_compute_metrics(results_dir, dataset)
%% Computes the AP and PQ for instance segmentation and panoptic segmentation
%% respectively.
    if nargin < 2
        dataset = 'voc2012';
    end

    mAP_opts = get_mAP_options(dataset);
    results = compute_metrics(results_dir, mAP_opts);
    save(fullfile(results_dir, 'instance_segmentation_results.mat'), 'results');
end