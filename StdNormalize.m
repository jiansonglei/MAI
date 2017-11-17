function [ numFeature ] = StdNormalize( numFeature, dim )
%NORMALIZE 此处显示有关此函数的摘要
%   此处显示详细说明
if nargin < 2, dim = 1; end;

mu = mean(numFeature, dim);
std_v = std(numFeature, 0, dim);

numFeature = (numFeature - mu) ./ std_v;

end

