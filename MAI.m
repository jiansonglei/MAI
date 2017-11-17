%Data_vector is the final representation learned from MAI 
%Data_vector may be high-dimentional matrix. 
%Dimension reduction can help to reduce the dimension of representation matrix

load('ACA.mat'); %ACA data consists of categorical data(Cate_data) and continuous data(Conti_data)
load('Echo.mat')

k=length(unique(label));
num_object=size(Cate_data,1);
move_t=0;

lambda = 0.1;

%normalization = true;
normalization = 0;
[onehot_feat, nCat] = OnehotRep(Cate_data);
n_feat = Conti_data;
for i = 1:size(n_feat,2)
    nanidx = isnan(n_feat(:,i));
    if any(nanidx)
        n_feat(nanidx,i) = mean(n_feat(~nanidx,i));
    end
end

if normalization
    n_feat = StdNormalize(n_feat);
end

nc_feat = NCFeatures(Conti_data, onehot_feat, nCat, lambda, normalization);

matrix_1 = [n_feat, onehot_feat]; %plain feature
matrix_2 = nc_feat; % coupled feature



%% neural parts
adam_Wp = InitAdamParam;
adam_Wc = InitAdamParam;
adam_Lp = InitAdamParam;
adam_Lc = InitAdamParam;

nObject = num_object;

pfeat = [matrix_1, ones(nObject, 1)];
cfeat = [matrix_2, ones(nObject, 1)];
% pfeat = matrix_1;
% cfeat = matrix_2;

nphidunits = 200;
nchidunits = 200;
npdim = 60;
ncdim = 60;


npfeatures = size(pfeat,2);
ncfeatures = size(cfeat,2);

Wp = 0.1 * gpuArray.randn(npfeatures, nphidunits);
Wc = 0.1 * gpuArray.randn(ncfeatures, nchidunits);

Lp = 0.1 * gpuArray.randn(nphidunits, npdim);
Lc = 0.1 * gpuArray.randn(nchidunits, ncdim);

validSet = ceil(rand(5000, 3) * nObject);

lsFun = @EucleadLSFun;

wtDecay = 1e-4;
batchSize = 200;
nBatch = ceil(nObject / batchSize);
nPairs = 100;
MaxIt = 20;

logProp = gpuArray(zeros(MaxIt, 1));

start = tic;
for i=1:MaxIt
    featIdx = randperm(nObject);
    for p=1:nPairs
        for j = 1:nBatch
            idx = (j-1)*batchSize+1 : min(j*batchSize, nObject);
            triple = [featIdx(idx)', ceil(rand(length(idx), 2) * nObject)];
            [ Wp, Wc, Lp, Lc, adam_Wp, adam_Wc, adam_Lp, adam_Lc] = ...
                AutoAdapter( pfeat, cfeat, triple, Wp, Wc, Lp, Lc, ...
                adam_Wp, adam_Wc, adam_Lp, adam_Lc, wtDecay,lsFun); %EucleadLSFun or CosLSFun
        end
    end
    logProp(i) = ValidationProp( pfeat, cfeat, validSet, Wp, Wc, Lp, Lc, lsFun);
    
    dtm = toc(start);
    fprintf('\nEpoch %d/%d finished, est. in %g secs, validation: %g', i, MaxIt, (MaxIt-i)*dtm/i, logProp(i));
end

datarep = [ 1./(1 + exp(-pfeat * Wp)), 1./(1 + exp(-cfeat * Wc)) ];
Data_vector=gather(datarep);



