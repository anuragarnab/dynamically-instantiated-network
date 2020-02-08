% ------------------------------------------------------------------------ 
%  Copyright (C)
%  Torr Vision Group (TVG)
%  University of Oxford - United Kingdom
% 
%  Bernardino Romera Paredes, Anurag Arnab, Qizhu Li
%  November 2015
% ------------------------------------------------------------------------ 

function [all_overlaps, all_scores, gt_counter] = compute_overlaps(prediction_root, class_id, opts)
    %   compute_overlaps: Computes the IoU between ground truth and predicted segments.
    %   Args:
    %       prediction_root: The directory with all the predicted results
    %       class_id: The semantic class id for which to compute overlaps.
    %           The IoU between two segments is 0 if the class_id is not the same.
    %       opts: A structure of options.
    %   Returns
    %       all_overlaps: The maximum IoU of each predicted segment wrt a ground truth
    %           segment in the dataset.
    %       all_scores: The scores corresponding to the segment with the maximum IoU wrt
    %           to each ground truth segment.
    %       gt_counter: The number of ground truth instances
    
    do_convert_to_indexed_png = opts.convert_to_indexed_png;
    list_file = opts.list_file; 
    ignore_label = opts.ignore_label;
    ignore_enabled = opts.ignore_enabled;

    if (ignore_enabled && class_id == 1)
        fprintf('Ignoring pixels in prediction corresponding to ignore label in ground truth\n'); 
    end

    [filenames] = read_file(list_file);
    range = [1:length(filenames)];

    all_scores = [];
    all_overlaps = [];
    gt_counter = 0;
    it = 1;

    for it = range

        filename = filenames{it};
        [object_im, colour_map] = imread(fullfile(opts.annotation_instance_root, [filename, opts.anno_suffix]));
        class_im = imread(fullfile(opts.annotation_semantic_root, [filename, opts.anno_suffix]));

        prediction_im_path = fullfile(prediction_root, [filename, opts.pred_suffix]);
        prediction_im = imread(prediction_im_path);

        if (do_convert_to_indexed_png)
            prediction_im = rgb2ind(prediction_im, colour_map);
            prediction_im = uint8(prediction_im);
        end

        if ~ isequal(size(prediction_im) ,size(object_im) )
            error('Prediction and ground truth image sizes don''t match') 
        end

        prediction_scores_path = fullfile(prediction_root, [filename, opts.score_suffix]);
        pred_classes = [];

        temp = dir(prediction_scores_path);
        if (temp.bytes > 0)
            data = dlmread(prediction_scores_path);
            scores = data(:,2);
            pred_classes = data(:,1);
        else
            scores = [];
        end

        labels = unique(prediction_im);
        labels = sort(labels, 'ascend');
        labels(labels == opts.not_eval_label) = [];
        labels(labels == ignore_label) = [];    

        if (ignore_enabled)
            ignore_mask = (object_im == ignore_label);                
            prediction_im( ignore_mask ) = ignore_label;
        end

        elements = unique(object_im);

        class_of_interest_mask = zeros(size(scores));
        class_of_interest_mask(pred_classes == class_id) = 1;
        class_of_interest_mask = logical(class_of_interest_mask);

        labels = labels(class_of_interest_mask);
        scores = scores(class_of_interest_mask);
        pred_classes = pred_classes(class_of_interest_mask);
        overlaps = zeros(length(scores),1);

        for j=1:length(elements)
            class_el = class_im(object_im == elements(j));
            class_el = class_el(1);

            if class_el ~= class_id
                continue
            end

            gt_counter = gt_counter + 1;
            gt_mask = object_im==elements(j);

            for s=1:length(scores)
                if (pred_classes(s) ~= class_id)
                    fprintf('Should never enter this\n');
                    error('Exception'); 
                end;

                pred_mask = (prediction_im==labels(s));    

                val = IoU(gt_mask, pred_mask);
                overlaps(s) = max(overlaps(s), val);
            end
        end
        all_scores = [all_scores; scores];
        all_overlaps = [all_overlaps; overlaps];
    end
end

function [filenames] = read_file(list_file)
    fp = fopen(list_file, 'r');
    C = textscan(fp, '%s %*[^\n]');
    filenames = C{1};
    fclose(fp);
end
    