% ------------------------------------------------------------------------ 
%  Copyright (C)
%  Torr Vision Group (TVG)
%  University of Oxford - United Kingdom
% 
%  Bernardino Romera Paredes, Anurag Arnab, Qizhu Li
%  November 2015
% ------------------------------------------------------------------------ 

function val = IoU(a, b)
% IoU Computes the IoU between two binary matrices a and b.
    a = double(a(:));
    b = double(b(:));

    intersection = a'*b;
    val = intersection/(sum(a)+sum(b)-intersection);
end

