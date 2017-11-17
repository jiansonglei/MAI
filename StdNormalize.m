function [ numFeature ] = StdNormalize( numFeature, dim )
%NORMALIZE �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
if nargin < 2, dim = 1; end;

mu = mean(numFeature, dim);
std_v = std(numFeature, 0, dim);

numFeature = (numFeature - mu) ./ std_v;

end

