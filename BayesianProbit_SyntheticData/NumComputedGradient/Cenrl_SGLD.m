clear all
clc
close all

%%% Load Global Sample 
% load('/Users/liudongzhu/Desktop/MATLAB/Bayes MC/Paper/BayesianProbit/Synthetic Data/Global Sampling/Global_Sample_SyntheticData.mat')
load('C:\Users\huangkb\Desktop\DZ\VCMC\PaperResult\BayesianProbit\Synthetic Data\Global_Sample_SyntheticData.mat')
Cov_global=theta(:, burin+1:size(theta,2))*theta(:,burin+1:size(theta,2))'/length([burin+1:size(theta,2)]);
% Mean_global=mean(theta(:, burin+1:size(theta,2)),2); 
clear theta 

rng(0) 

Ns=50; 

nb=500; %500; %mini batch of data


N=size(X,1);
d=size(X,2);

MaxNumgrad=50*Ns*N;
MaxOutIter=MaxNumgrad/N; 







delta=0.2;
alpha=1e-2;
beta=1;
gamma=0.7;  %



for ave_index=1:100
ave_index
    tic 

sample_set=[];
theta_sgld=mvnrnd(zeros(d,1), sigma_prior^2*eye(d))';

% % gradient descent for Max likelihood, which is used for initial 
% for t=1:500
%    mid=theta_sgld'*X';
%    rev_normcdf=normcdf(mid);
%    rev_normcdf(find(rev_normcdf==1))=1-10^-10;
%    rev_normcdf(find(rev_normcdf==0))=0+10^-10;
%    
%    grad = -theta_sgld/(sigma_prior^2)+ (normpdf(mid).* (Y'- normcdf(mid))./ (rev_normcdf.*(1-rev_normcdf)) *X)';
%    
% %    recod_conv(t)=norm(grad); 
%    
%    theta_sgld=theta_sgld+0.00001*grad;   
% %    acc(t)=sum(abs((sign(X*theta0)+1)/2-Y))/size(Y,1);
% end

t_mix=1e4;


for outer_loop=1:MaxOutIter
    
     index=randperm(size(X,1));
     X=X(index,:);
     Y=Y(index); 
    
     for m=1:floor(N/nb)
                
        if m<floor(size(X,1)/nb) 
            X_batch= X((m-1)*nb+1:m*nb,:); 
            Y_batch=Y((m-1)*nb+1:m*nb); 
        else
            X_batch= X((m-1)*nb+1:size(X,1),:); 
            Y_batch=Y((m-1)*nb+1:size(X,1)); 
        end
        mid=theta_sgld'*X_batch';
        rev_normcdf=normcdf(mid);
        rev_normcdf(find(rev_normcdf==1))=1-10^-10;
        rev_normcdf(find(rev_normcdf==0))=0+10^-10;

        t=(outer_loop-1)*floor(N/nb)+m;
        
        Grad_theta = -theta_sgld/(sigma_prior^2)+ N/nb* (normpdf(mid).* (Y_batch'- normcdf(mid))./ (rev_normcdf.*(1-rev_normcdf)) *X_batch)';

        eps_sgld= alpha*(beta+t)^(-gamma); 
        theta_sgld= theta_sgld + eps_sgld*Grad_theta + sqrt(2*eps_sgld)*randn(d,1);
    
        sample_set=[sample_set,theta_sgld];

        num_cur=size(sample_set,2);
        
             
          if t>=t_mix+Ns
             Err_cov_SGLD_all(ave_index,t)= sum(sum(abs(sample_set(:, t_mix:num_cur)*sample_set(:, t_mix:num_cur)'/(num_cur-t_mix+1)-Cov_global)./abs(Cov_global)))/d/d;
          end
    
     end
     
     if t>=t_mix+Ns
     figure(1)
     plot([t_mix+Ns:t]*nb,Err_cov_SGLD_all(ave_index,t_mix+Ns:t))

     pause(0.01)
     end

end

toc

end
% 
% clear Err_cov_SGLD
% for t=2*Nte_sgld: 100: MaxNumgrad
%       Err_cov_SGLD(t)= sum(sum(abs(sample_set(:,t-Nte_sgld+1:t)*sample_set(:,t-Nte_sgld+1:t)'/Nte_sgld-Cov_global)./abs(Cov_global)))/d/d;
% end