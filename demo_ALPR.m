% % % The code is written by Jie Wen, if you have any problems, 
% % % please don't hesitate to contact me: jiewen@hrbeu.edu.cn 
 
% % % If you find the code is useful, please cite the following reference:
% [1] Wen J, Zhong Z, Zhang Z, Fei L, Lai Z, Chen R, 
% Adaptive Locality Preserving Regression [J], 
% IEEE Transactions on Circuits and Systems for Video Technology, 2019
% [2] Wen J, Fei L, Lai Z, Zhang Z, Wu J, Fang X, 
% Adaptive Locality Preserving based Discriminative Regression [C]. 
% International Conference on Pattern Recognition, 2018: 535-540.
clc,clear;
clc
clear memory;

name = 'LFW_1024_jerry';
load (name);
fea = double(fea);
sele_num = [7];

nnClass = length(unique(gnd));
num_Class = [];
for i = 1:nnClass
    num_Class = [num_Class length(find(gnd==i))];
end

Train_Ma  = [];
Train_Lab = [];
Test_Ma   = [];
Test_Lab  = [];
for j = 1:nnClass    
    idx = find(gnd==j);
    randIdx  = randperm(num_Class(j));
    Train_Ma = [Train_Ma;fea(idx(randIdx(1:sele_num)),:)];            
    Train_Lab= [Train_Lab;gnd(idx(randIdx(1:sele_num)))];
    Test_Ma  = [Test_Ma;fea(idx(randIdx(sele_num+1:num_Class(j))),:)]; 
    Test_Lab = [Test_Lab;gnd(idx(randIdx(sele_num+1:num_Class(j))))];
end
Train_Ma = Train_Ma';   
Train_Ma = Train_Ma./repmat(sqrt(sum(Train_Ma.^2)),[size(Train_Ma,1) 1]);
Test_Ma  = Test_Ma';
Test_Ma  = Test_Ma./repmat(sqrt(sum(Test_Ma.^2)),[size(Test_Ma,1) 1]); 

X = Train_Ma;
lambda1 = 0.1;
lambda2 = 0.1;
MaxIter = 40;  

[W,T,S,obj] = ALPR(X,Train_Lab,nnClass,lambda1,lambda2,MaxIter);
aa = sum(W.*W,2);
rho = 0.0001;
[index] = find(aa<rho);
W(index,:) = 0;

Train_Maa = W'*Train_Ma;
Test_Maa  = W'*Test_Ma;
Train_Maa = Train_Maa./repmat(sqrt(sum(Train_Maa.^2)),[size(Train_Maa,1) 1]);
Test_Maa  = Test_Maa./repmat(sqrt(sum(Test_Maa.^2)),[size(Test_Maa,1) 1]);    
[class_test] = knnclassify(Test_Maa', Train_Maa', Train_Lab,1,'euclidean','nearest');
rate_KNN     = sum(Test_Lab == class_test)/length(Test_Lab)*100

