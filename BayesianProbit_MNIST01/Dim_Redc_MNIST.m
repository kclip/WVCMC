clear all
clc
rng(0)


load('~Location\MNIST.mat')

index1=find(train_set(:,1)==1);
index0=find(train_set(:,1)==0);

X=train_set([index1;index0],2:785);
Y=train_set([index1;index0],1);

sub=pca(X,'NumComponents',30); 

X=X*sub;

% X_tsne=tsne(X);
% 
% gscatter(X_tsne(:,1),X_tsne(:,2),Y);


d=size(X,2); 

N=size(X,1); 
