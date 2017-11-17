function [ onehots, nCat ] = OnehotRep( categorydata )
nCat = max(categorydata, [], 1);
uniqueIdx = cumsum(nCat);
maxC = uniqueIdx(end);

uniqueIdx = [0, uniqueIdx(1:end-1)];

m = size(categorydata, 1);
categorydata = categorydata +  uniqueIdx;

onehots = zeros(m, maxC);

for i=1:m
    onehots(i, categorydata(i, :)) = 1;
end

end

