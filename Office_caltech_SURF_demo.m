clear;clc;
warning off;
global options
% Set algorithm parameters 
options.r = 1.5; 
options.eta = 0.5; 
options.lambda = 10;
options.T = 10; 

srcStr = {'amazon','amazon','amazon','Caltech10','Caltech10','Caltech10','dslr','dslr','dslr','webcam','webcam','webcam'};
tgtStr = {'Caltech10','dslr','webcam','amazon','dslr','webcam','amazon','Caltech10','webcam','amazon','Caltech10','dslr'};

ffid = fopen('result_office_caltech_surf.txt','at');
fprintf(ffid, '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n %s\n', datestr(now));
fprintf(ffid, 'r = %.1f  eta = %.1f lambda = %.1f\n',options.r,options.eta,options.lambda);
datapath = 'D:\Program Files\MATLAB\R2020b\bin\lan\DA-1\data\Office+Caltech+surf\';
acc_iter = [];
all_acc = [];
for iData = 1:12
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    options.data = strcat(src,'_vs_',tgt);

    % Preprocess data using L2-norm
    load(fullfile(datapath,[src,'_SURF_L10.mat']));
    X_src = normr(fts);
    X_src = zscore(X_src,1)';
    Y_src = labels;
    load(fullfile(datapath,[tgt,'_SURF_L10.mat']));
    X_tar = normr(fts);
    X_tar = zscore(X_tar,1)';
    Y_tar = labels;
    
    fprintf('$$$$$$$$$$$$$$$ --%s-- $$$$$$$$$$$$$$\n' ,options.data);
    
    %%
    [acc,acc_ite,~] = DLAD(X_src,Y_src,X_tar,Y_tar);
    ACCi(iData)=acc;
    acc_iter = [acc_iter;acc_ite'];
    acc = 100*acc;
    all_acc = [all_acc acc];
    fprintf('******************************\n%s :\naccuracy: %.2f\n\n',options.data,acc);
    fprintf(ffid,'***************************\n%s :\naccuracy: ',options.data);
    fprintf(ffid,'%.2f\n', acc);
end
fprintf('%.2f\n', mean(all_acc));
fclose(ffid);