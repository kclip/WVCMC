clear all
clc
close all


%%% Load Global Sample 
load('C:\Users\huangkb\Desktop\DZ\VCMC\PaperResult\Generate Sample DimRedc MNIST\GlobalSample_MNIST_01_Dim30.mat')
% load('/Users/liudongzhu/Desktop/MATLAB/Bayes MC/Paper/BayesianProbit/Synthetic Data/Global Sampling/Global_Sample_SyntheticData.mat')
Cov_global=theta(:, burin+1:size(theta,2))*theta(:,burin+1:size(theta,2))'/length([burin+1:size(theta,2)]);
Mean_global=mean(theta(:, burin+1:size(theta,2)),2); 
clearvars -except thetaLocal_set Cov_global Mean_global theta X Y sigma_prior K N

rng(0) 










parfor ave_index=1:100

tic 

SingleExperiment_SGLD(X,Y,sigma_prior,ave_index,Cov_global); %theta column vector

toc

end
% 
% clear Err_cov_SGLD
% for t=2*Nte_sgld: 100: MaxNumgrad
%       Err_cov_SGLD(t)= sum(sum(abs(sample_set(:,t-Nte_sgld+1:t)*sample_set(:,t-Nte_sgld+1:t)'/Nte_sgld-Cov_global)./abs(Cov_global)))/d/d;
% end