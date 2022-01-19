clear;clear global;clc;
warning off;
global options
% Set algorithm parameters
options.r = 1.3;
options.eta = 0.5; 
options.lambda = 10;
options.T = 10;  

srcStr = {'Art_Art','Art_Art','Art_Art','Clipart_Clipart','Clipart_Clipart','Clipart_Clipart',...
    'Product_Product','Product_Product','Product_Product','RealWorld_RealWorld','RealWorld_RealWorld','RealWorld_RealWorld'};
tgtStr = {'Art_Clipart','Art_Product','Art_RealWorld','Clipart_Art','Clipart_Product','Clipart_RealWorld',...
    'Product_Art','Product_Clipart','Product_RealWorld','RealWorld_Art','RealWorld_Clipart','RealWorld_Product'};

ffid = fopen('result_office_home_resnet.txt','at');
fprintf(ffid, '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n %s\n', datestr(now));
fprintf(ffid, 'r = %.1f  eta = %.1f lambda = %.1f\n',options.r,options.eta,options.lambda);
datapath = 'D:\Program Files\MATLAB\R2020b\bin\lan\DA-1\data\Office-Home_resnet50\';
all_acc = [];
for iData = 1:12
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    options.data = strcat(src,'_vs_',tgt);
    
    % Preprocess data using L2-norm
    load(fullfile(datapath,[src,'.mat']));
    X_src = normc(fts);
    X_src = zscore(X_src',1)';
    Y_src = labels;
    load(fullfile(datapath,[tgt,'.mat']));
    X_tar = normc(fts);
    X_tar = zscore(X_tar',1)';
    Y_tar = labels;
    
    fprintf('$$$$$$$$$$$$$$$ --%s-- $$$$$$$$$$$$$$\n' ,options.data);
    
    %%
    [acc,acc_ite,~] = DLAD(X_src,Y_src,X_tar,Y_tar);
    ACCi(iData)=acc;
    acc = 100*acc;
    all_acc = [all_acc acc];
    fprintf('******************************\n%s :\naccuracy: %.2f\n\n',options.data,acc);
    fprintf(ffid,'******************************\n%s :\naccuracy: ',options.data);
    fprintf(ffid,'%.2f\n', acc);
end
fprintf('%.2f\n', mean(all_acc));
fclose(ffid);