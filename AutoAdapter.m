function [ Wp, Wc, Lp, Lc, adam_Wp, adam_Wc, adam_Lp, adam_Lc] = ...
    AutoAdapter( feat1, feat2, triple, Wp, Wc, Lp, Lc, ...
    adam_Wp, adam_Wc, adam_Lp, adam_Lc, weightDecay, lsFun)

%  nfeat = [batch_size, nhid]
%  cfeat = [batch_size, chid]

% nonlinear transform
plainF = feat1(triple(:,1), :);
plainF1 = feat1(triple(:,2), :);
plainF2 = feat1(triple(:,3), :);

hidP = 1./(1 + exp(-plainF * Wp));
hidP1 = 1./(1 + exp(-plainF1 * Wp));
hidP2 = 1./(1 + exp(-plainF2 * Wp));

jointF = feat2(triple(:,1), :);
jointF1 = feat2(triple(:,2), :);
jointF2 = feat2(triple(:,3), :);

hidC = 1./(1 + exp(-jointF * Wc));
hidC1 = 1./(1 + exp(-jointF1 * Wc));
hidC2 = 1./(1 + exp(-jointF2 * Wc));

% delta_hid_1 = hidP - hidP1;
% delta_hid_2 = hidP - hidP2;
% deltaDistP = sum((delta_hid_1 * Lp).^2, 2) - sum((delta_hid_2 * Lp).^2, 2);
deltaDistP = lsFun(hidP, hidP1, hidP2);


% delta_hid_1 = hidC - hidC1;
% delta_hid_2 = hidC - hidC2;
% deltaDistC = sum((delta_hid_1 * Lc).^2, 2) - sum((delta_hid_2 * Lc).^2, 2);
deltaDistC = lsFun(hidC, hidC1, hidC2);

pNeg = deltaDistP <= 0;
cNeg = deltaDistC <= 0;

% negDist = pNeg & cNeg; % agree
% 
% disagree = pNeg ~= cNeg;
% rnd = rand(nnz(disagree),1) > .5;
% pChoice = disagree;
% pChoice(disagree) = rnd;
% cChoice = disagree;
% cChoice(disagree) = ~rnd;
% 
% negDist(pChoice & pNeg) = 1;
% negDist(cChoice & cNeg) = 1;

negDist = pNeg;
temp = jointF1(negDist,:);
jointF1(negDist,:) = jointF2(negDist,:);
jointF2(negDist,:) = temp;

[ Wc, Lc, adam_Wc, adam_Lc] = LearnSinglePart( jointF, jointF1, jointF2, Wc, Lc, adam_Wc, adam_Lc, weightDecay);

negDist = cNeg;
temp = plainF1(negDist,:);
plainF1(negDist,:) = plainF2(negDist,:);
plainF2(negDist,:) = temp;

[ Wp, Lp, adam_Wp, adam_Lp] = LearnSinglePart( plainF, plainF1, plainF2, Wp, Lp, adam_Wp, adam_Lp, weightDecay);

end

function [ W, L, adam_W, adam_L] = LearnSinglePart( feat, feat1, feat2, W, L, adam_W, adam_L, weightDecay)

hid = 1./(1 + exp(-feat * W));
hid1 = 1./(1 + exp(-feat1 * W));
hid2 = 1./(1 + exp(-feat2 * W));
batch_size = size(feat,1);

delta_hid_1 = (hid - hid1);
delta_hid_2 = (hid - hid2);

% D1 = ||(chid - nn1Chid)*L1||_2 = delta_chid_1*L1*L1'*delta_chid_1'
D1 = sum((delta_hid_1 * L).^2, 2);
% D2 = ||(chid - nn2Chid)*L1||_2 = delta_chid_2*L1*L1'*delta_chid_2'
D2 = sum((delta_hid_2 * L).^2, 2);

z = D1 - D2;

% Loss = log(logistic(z))
% d_log(logistic(z)) = 1/(1+exp(z))
d_Loss = 1./(1+exp(z));

%% Gradient L
% d_z / d_L = 2 * (delta_chid_1*delta_chid_1' - delta_chid_2*delta_chid_2') *  L
d_L = 2 * ( (delta_hid_1 .* d_Loss)' * delta_hid_1 - (delta_hid_2 .* d_Loss)' * delta_hid_2 ) *  L;
mean_d_L = d_L ./ batch_size - weightDecay * L;


M = L * L';
%% Gradient W
d_hid = ((hid2 - hid1) * M) .* (hid .* (1-hid));
d_hid1 = (delta_hid_1 * M) .* (hid1 .* (1-hid1));
d_hid2 = (delta_hid_2 * M) .* (hid2 .* (1-hid2));
d_W = 2 .* ( (feat .* d_Loss)' * d_hid  - (feat1 .* d_Loss)' * d_hid1 - (feat2 .* d_Loss)' * d_hid2 );
mean_d_W = d_W ./ batch_size - weightDecay * W;

%% Update parameters
[grad, adam_L] = AdamUpdate(mean_d_L, adam_L);
L = L + grad;

[grad, adam_W] = AdamUpdate(mean_d_W, adam_W);
W = W + grad;

end
