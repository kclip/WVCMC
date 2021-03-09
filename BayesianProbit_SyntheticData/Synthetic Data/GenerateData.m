clear all
clc
close all

rng(1) % For reproducibility
N = 8500; % num samples
d = 5; % data dimension

X=randn(N,d);


%w=ones(1,d);
sigma_prior=1;  %%%% prior variance    
theta_tru= mvnrnd(zeros(d,1),eye(d)*sigma_prior^2);



Y = (sign( X*theta_tru' + randn(N,1))+1)/2;