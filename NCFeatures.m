function [ ncfeat ] = NCFeatures( numFeature, onehots, nCat, lambda, normalize )

[m,n] = size(onehots);
ncfeat = zeros(m, n * size(numFeature, 2));

lambda_vec = zeros(1, n);
cumCat = cumsum(nCat);

start = 1;
for i = 1:length(nCat)
    lambda_vec(start:cumCat(i)) = lambda / (nCat(i) - 1);
    start = cumCat(i) + 1;
end

for j = 1:size(numFeature, 2)
    offset = (j - 1) * n;
    feat = numFeature(:,j);
    vldIdx = ~isnan(feat);
    
    for i = 1:n
        valIdx = onehots(:,i) > 0;
        vals = feat(valIdx & vldIdx);
        mu = mean(vals);
        if isnan(mu), mu = 0; end
        
        nvals = feat(~valIdx & vldIdx);
        allvals = feat(vldIdx);
        nmu = mean(allvals);
        if isnan(nmu), nmu = 0; end
        
        if normalize  
            std_v = std(allvals);
            if std_v == 0 || isnan(std_v)
                std_v = 1;
            end
            
            vals = (vals - nmu) ./ std_v;
            
            mu = 0;
            % normalize non this category data
            
            std_v = std(allvals);
            if std_v == 0 || isnan(std_v)
                std_v = 1;
            end
            nvals = (nvals - nmu) ./ std_v;
            
            nmu = 0;
        end
        
        ncfeat(valIdx & vldIdx, offset + i) = vals .* (1-lambda_vec(i));
        ncfeat(valIdx & ~vldIdx, offset + i) = nmu .* (1-lambda_vec(i));
        
        ncfeat(~valIdx & vldIdx, offset + i) = nvals .* lambda_vec(i);
        ncfeat(~valIdx & ~vldIdx, offset + i) = nmu .* lambda_vec(i);
    end
end

end

