clear all
clc
close all
 
 
rng(0)

load('~Loacation\BayesianProbit_SyntheticData\Synthetic Data\SyntheticData.mat')



Ns_use=1000;
 
burin=100;
interval=10;
 
index1=find(Y==1);
index0=find(Y==0);

X1=X(index1,:); 
Y1=Y(index1); 
X0=X(index0,:); 
Y0=Y(index0);  
clear X Y 


K=10; 


for sk=0:0.5:3
    distr=[1:K].^(-sk)/(sum([1:K].^(-sk))) ; 
    sample_size1=floor(distr*size(X1,1)); 
    sample_size0=floor(fliplr(distr)*size(X0,1));
    
    sample_index1E=cumsum(sample_size1);
    sample_index1S=sample_index1E-sample_size1+1;
    sample_index0E=cumsum(sample_size0);
    sample_index0S=sample_index0E-sample_size0+1;

    tic
    parfor k = 1:K


        X_k=[X1(sample_index1S(k):sample_index1E(k),:);X0(sample_index0S(k):sample_index0E(k),:)];
        Y_k=[Y1(sample_index1S(k):sample_index1E(k));Y0(sample_index0S(k):sample_index0E(k))];


        GibbsSampingFunc(X_k,Y_k,sigma_prior*sqrt(K),Ns_use,interval, burin,k,sk);


    end
    toc

end
