function [Acc,acc_iter,Yt_pred] = DLAD(X_src,Y_src,X_tar,Y_tar)
global options
[m,ns] = size(X_src);
nt = size(X_tar,2);
Ys = bsxfun(@eq, Y_src(:), 1:max(Y_src));
Ys = Ys';
n = ns+nt;
C = size(Ys,1);
X = [X_src X_tar];
X = X*diag(sparse(1./sqrt(sum(X.^2))));

% Construct kernel
K = kernel_meda('rbf',X,sqrt(sum(sum(X .^ 2).^0.5)/(n + m)));
Ks = K(:,1:ns);
Kt = K(:,ns+1:end);
Beta = (Ks*Ks'+0.1*eye(n))\(Ks*Ys');
Yt_pred = Beta'*Kt;
[~,Cls] = max(Yt_pred',[],2);
acc = mean(Y_tar == Cls);

T = eye(C); %label encoding matrix

for c = 1:C
    nsc(c) = length(find(Y_src == c));
end

acc_iter = [];
for iter = 1:options.T
    % Compute coefficients vector W and Ypred via SLR
    % Compute P
    for j=1:C
        v = Yt_pred-repmat(T(:,j),1,nt);
        q(:,j)=sum(v'.*v',2);
    end
    if options.r==1
        P = zeros(nt,C);
        [~,idx] = min(q,[],2);
        for i = 1:nt
            P(i,idx(i)) = 1;
        end
    else
        %         mm = (options.r.*q).^(1/(1-options.r-eps));
        %         s0 = sum(mm,2);
        %         P = mm./repmat(s0,1,C);
        P = q.^(1/(1-options.r-eps))./repmat(sum(q.^(1/(1-options.r-eps)),2),1,C);
    end
    F = (P.^options.r)';
    S = diag(sum(F,1));
    
    % Construct Hard MMD matrix
%     [~,ClsCls] = max(P,[],2);
%     PP = bsxfun(@eq, ClsCls(:), 1:max(ClsCls));
%     ntc = sum(PP,1);
%     e = [1 / ns * ones(ns,1); -1 / nt * ones(nt,1)];
%     G1 = e * e' * C;
%     Ns = diag(nsc)^-1;
%     Nt = diag(ntc)^-1;
%     G2 = [Ys'*(Ns*Ns)*Ys,-Ys'*(Ns*Nt)*PP';-PP*(Nt*Ns)*Ys,PP*(Nt*Nt)*PP'];
%     G = G1 + G2;
%     G = G / norm(G,'fro');
    
    % Construct Soft MMD matrix
    ntc = sum(P,1);
    e = [1 / ns * ones(ns,1); -1 / nt * ones(nt,1)];
    G1 = e * e' * C;
    Ns = diag(nsc)^-1;
    Nt = diag(ntc)^-1;
    G2 = [Ys'*(Ns*Ns)*Ys,-Ys'*(Ns*Nt)*P';-P*(Nt*Ns)*Ys,P*(Nt*Nt)*P'];
    G = G1 + G2;
    G = G / norm(G,'fro');
    
    
    %Compute Beta
    A = blkdiag(eye(ns),S^0.5);
    Y = [Ys,F*S^-1];
    Beta = ((A * A + options.lambda * G ) * K + options.eta * speye(n,n)) \ ( A * A * Y');
    Yt_pred = Beta' * Kt;
    [~,Cls] = max(Yt_pred',[],2);
    
    %% Compute accuracy
    Acc = mean(Y_tar == Cls);
    acc_iter = [acc_iter;Acc];
    %fprintf('Iteration:%02d, Acc=%f\n',iter,Acc);
end
end