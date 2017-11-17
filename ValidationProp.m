function [ logProp ] = ValidationProp( feat1, feat2, triple, Wp, Wc, Lp, Lc, lsFun)


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

deltaDistP = lsFun(hidP, hidP1, hidP2);
deltaDistC = lsFun(hidC, hidC1, hidC2);

logProp = sum((deltaDistP < 0) == (deltaDistC < 0)) / length(deltaDistP);

% negDist = deltaDistP < 0;
% temp = hidC1(negDist,:);
% hidC1(negDist,:) = hidC2(negDist,:);
% hidC2(negDist,:) = temp;
% 
% negDist = deltaDistC < 0;
% temp = hidP1(negDist,:);
% hidP1(negDist,:) = hidP2(negDist,:);
% hidP2(negDist,:) = temp;
% 
% delta_hid_1 = (hidP - hidP1);
% delta_hid_2 = (hidP - hidP2);
% 
% % D1 = ||(chid - nn1Chid)*L1||_2 = delta_chid_1*L1*L1'*delta_chid_1'
% D1 = sum((delta_hid_1 * Lp).^2, 2);
% % D2 = ||(chid - nn2Chid)*L1||_2 = delta_chid_2*L1*L1'*delta_chid_2'
% D2 = sum((delta_hid_2 * Lp).^2, 2);
% 
% logProp = sum(-log(1+exp(-abs(D1 - D2))));
% 
% 
% delta_hid_1 = (hidC - hidC1);
% delta_hid_2 = (hidC - hidC2);
% 
% % D1 = ||(chid - nn1Chid)*L1||_2 = delta_chid_1*L1*L1'*delta_chid_1'
% D1 = sum((delta_hid_1 * Lc).^2, 2);
% % D2 = ||(chid - nn2Chid)*L1||_2 = delta_chid_2*L1*L1'*delta_chid_2'
% D2 = sum((delta_hid_2 * Lc).^2, 2);
% 
% logProp = logProp + sum(-log(1+exp(-abs(D1 - D2))));

end

