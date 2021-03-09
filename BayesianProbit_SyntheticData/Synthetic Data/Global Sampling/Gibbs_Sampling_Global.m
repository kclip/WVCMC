clear all
clc
close all

rng(1)

load('~Loacation\BayesianProbit_SyntheticData\Synthetic Data\SyntheticData.mat')


burin=100;

theta0=mvnrnd(zeros(d,1), sigma_prior^2*eye(d))';

% gradient descent for Max likelihood, which is used for initial 
for t=1:500
   mid=theta0'*X';
   rev_normcdf=normcdf(mid);
   rev_normcdf(find(rev_normcdf==1))=1-10^-10;
   rev_normcdf(find(rev_normcdf==0))=0+10^-10;
   
   grad = -theta0/(sigma_prior^2)+ (normpdf(mid).* (Y'- normcdf(mid))./ (rev_normcdf.*(1-rev_normcdf)) *X)';
   
%    recod_conv(t)=norm(grad); 
   
   theta0=theta0+0.00001*grad;   
%    acc(t)=sum(abs((sign(X*theta0)+1)/2-Y))/size(Y,1);
end

% figure(1)
% semilogy(recod_conv)
% figure(2)
% plot(acc) 

Ng=20000;

C=inv( (X'*X)+eye(d)/sigma_prior^2 );
theta=zeros(d,Ng+burin);
theta=[theta0,theta];


% 
% CLocal=zeros(d,d,K);
% thetaLocal=zeros(d,Ns+1,K);
% for k=1:K
%     CLocal(:,:,k)=inv( (X((k-1)*Nk+1:k*Nk,:)'*X((k-1)*Nk+1:k*Nk,:))+eye(d)/sigma_prior^2/K );
%     thetaLocal(:,1,k)=inv(X((k-1)*Nk+1:k*Nk,:)'*X((k-1)*Nk+1:k*Nk,:))*X((k-1)*Nk+1:k*Nk,:)'*Y((k-1)*Nk+1:k*Nk);
% end
% 
% 


for s=1: (Ng+burin)
  
    
    
    %%%%% Global Sampling for benchmark %%%%%%

    s
    
    tic
    nu=zeros(N,1);
    for i=1:N
        pd = makedist('Normal',theta(:,s)'*X(i,:)',1);     % Normal(mu,sigma)
        if Y(i)>0
            pdt = truncate(pd,0,inf);              % truncated to interval (0,1)
        else
            pdt = truncate(pd,-inf,0);
        end
        
        nu(i) = random(pdt,1,1);
    end
    
    theta(:,s+1)=mvnrnd(C*X'*nu,(C+C')/2,1)';
    
    toc 
  
    
    
end