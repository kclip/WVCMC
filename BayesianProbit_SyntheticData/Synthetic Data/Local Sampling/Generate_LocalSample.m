clear all
clc
close all
 
 
rng(0)

load('~Loacation\BayesianProbit_SyntheticData\Synthetic Data\SyntheticData.mat')

K=50;  
 
% 
% index=randperm(size(X,1));
% 
% X=X(index,:);
% Y=Y(index,:);
%  
 
Ns_use=1000;
 
burin=100;
interval=10;
 
 

  
loc=round(size(X,1)/K);

tic
parfor k = 1:K
    
    if k<K
        X_k=X((k-1)*loc+1:k*loc,:);
        Y_k=Y((k-1)*loc+1:k*loc);
    else
        X_k=X((k-1)*loc+1:size(X,1),:);
        Y_k=Y((k-1)*loc+1:size(X,1));
    end
    
   
    GibbsSampingFunc(X_k,Y_k,sigma_prior*sqrt(K),Ns_use,interval, burin,k);
    

end
toc

