clear all
clc
close all
 
 
rng(1)

load('~Location\BayesianProbit_MNIST01\MNIST_01_Dim30.mat')

K=10;

N=size(X,1);
d=size(X,2);
sigma_prior=1; 

index=randperm(size(X,1));

X=X(index,:);
Y=Y(index,:);
 
 
Ns_use=2000;
 
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

