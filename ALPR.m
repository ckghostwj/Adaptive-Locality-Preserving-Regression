function [W,T,S,obj] = ALPR(X,Train_Lab,nnClass,lambda1,lambda2,MaxIter)
% % % The code is written by Jie Wen, if you have any problems, 
% % % please don't hesitate to contact me: jiewen@hrbeu.edu.cn 
 
% % % If you find the code is useful, please cite the following reference:
% [1] Wen J, Zhong Z, Zhang Z, Fei L, Lai Z, Chen R, 
% Adaptive Locality Preserving Regression [J], 
% IEEE Transactions on Circuits and Systems for Video Technology, 2019
% [2] Wen J, Fei L, Lai Z, Zhang Z, Wu J, Fang X, 
% Adaptive Locality Preserving based Discriminative Regression [C]. 
% International Conference on Pattern Recognition, 2018: 535-540.

[m,n] = size(X);
W = rand(m,nnClass);
label = unique(Train_Lab);
T  = bsxfun(@eq, Train_Lab, label');
T  = double(T);
S1 = T*T';                        
S1 = S1-diag(diag(S1));           
D  = repmat(sum(S1,2),1,n);
S  = S1./(D+eps);

D = diag(1./(sqrt(sum(W.*W,2))+eps));
for iter = 1:MaxIter
    W_old = W;
    S_old = S;
    T_old = T;
    % ------- update W -------- %
    Sw = zeros(m,m);
    for i = 1:nnClass
        index = find(Train_Lab==i);
        sub_S = S(index,index);
        Xclass = X(:,index);
        A = sub_S.^2;
        D1 = diag(sum(A,2));
        D2 = diag(sum(A,1));
        L = D1+D2-A-A';
        Sw = Sw + length(index)*Xclass*L*Xclass';
    end     
    W = (X*X'+lambda1*Sw+0.5*lambda2*D)\(X*T);
    D = diag(1./max(sqrt(sum(W.*W,2)),1e-10)); 
    % ------------ update T ---------------- %
    R = W'*X;
    T1 = zeros(nnClass,n);
    for ind = 1:length(Train_Lab)
         T1(:,ind) = (optimize_R(R(:,ind)', Train_Lab(ind)))';
    end
    T = T1';
    % ----------- update S ------------- %
    WX = W'*X;
    distance_WX = L2_distance_1(WX,WX);
    distance_WX = 1./(distance_WX+eps);
    Dis_WX = distance_WX.*S1;
    S = Dis_WX.*repmat(1./max(sum(Dis_WX,2),1e-10),1,n); 
    
    L1 = norm(W-W_old,'fro')^2;
    L2 = norm(S-S_old,'fro')^2;
    L3 = norm(T-T_old,'fro')^2;
    obj(iter) = (norm(T-X'*W,'fro')^2+lambda1*trace(W'*Sw*W)+lambda2*sum(sqrt(sum(W.^2,2))))/norm(X,'fro')^2;    
    if iter > 2 && abs(obj(iter)-obj(iter-1)) < 1e-7
        iter
        break;
    end
end
end