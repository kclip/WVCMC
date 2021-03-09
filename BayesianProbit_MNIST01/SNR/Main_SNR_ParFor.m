clear all
clc
close all


rng(0)




%%% Load Local Sample 
K=10; 
for k=1:K
    path_name=['~Location\Generate Sample DimRedc MNIST\Local Sampling\MNIST01_Dim30_LocalSample_K10_k=',num2str(k),'.mat'];
 
    load(path_name)
    thetaLocal_set(:,:,k)=theta;
end
clear theta 



%%% Load Global Sample 
load('~Location\Generate Sample DimRedc MNIST\GlobalSample_MNIST_01_Dim30.mat')


Cov_global=theta(:, burin+1:size(theta,2))*theta(:,burin+1:size(theta,2))'/length([burin+1:size(theta,2)]);
Mean_global=mean(theta(:, burin+1:size(theta,2)),2); 
clearvars -except thetaLocal_set Cov_global Mean_global theta X Y sigma_prior K

N=size(X,1);
% X=[ones(N,1),X];
d=size(X,2);
 
 
mt=2*d+2;
mr=2*d;

 
Ns=50; 
Ns_NOMA= Ns*K;   %comm blocks=1000


interval=floor(size(thetaLocal_set,2)/Ns_NOMA); 
max_real=40/interval;  %channel realizations 
 
snr_set=[5:5:40];  %[10:5:50]; 

P=mr;   % rx poower 



for shuffle_time=1:max_real
    
  %random shuffle locally generated samples  
   index=randperm(size(thetaLocal_set,2));
   thetaLocal_set=thetaLocal_set(:,index,:); 

   
    parfor split_index=1:interval 
    
    thetaLocal=thetaLocal_set(:,split_index:interval:Ns*interval,:);
    thetaLocal_NOMA=thetaLocal_set(:,split_index:interval:Ns_NOMA*interval,:);
    
    
   
    

 
    Single_Expriment_SNR(X,Y,thetaLocal,thetaLocal_NOMA, shuffle_time, interval,split_index,K,d,mr,snr_set,P,Ns,Ns_NOMA,Cov_global,N,sigma_prior);
    
   

% %     (shuffle_time-1)*interval+split_index
% % %     
% % %     figure(1)
% % %     err_single=reshape(mean(Err_cov_SingleWorker,1),length(snr_set),K); 
% % %     semilogy(snr_set, min(err_single'),'-.','LineWidth',2,'Color',[0.4940    0.1840    0.5560])
% % %     hold on 
% % %     semilogy(snr_set,mean(Err_cov_GaussOMA,1),':','LineWidth',2,'Color',[0.9290    0.6940    0.1250])
% % %     hold on 
% % %     semilogy(snr_set,mean(Err_cov_ana,1),'--d','LineWidth',2,'Color',[ 0.8500    0.3250    0.0980]);
% % %     hold on 
% % %     semilogy(snr_set,mean(Err_cov_wgcmcN,1),'--*','LineWidth',2,'Color',[ 0    0.4470    0.7410]);
% % %     hold on
% % %     semilogy(snr_set,mean(Err_cov_GD_OMA,1),'-d','LineWidth',2,'Color',[ 0.8500    0.3250    0.0980]);
% % %     hold on 
% % %     semilogy(snr_set,mean(Err_cov_GD_NOMA,1),'-*','LineWidth',2,'Color',[ 0    0.4470    0.7410]);   
% % %     hold off
% % %     legend('Best Single Worker','GCMC','WGCMC (OMA)','WGCMC (NOMA)','WVCMC (OMA)','WVCMC (NOMA)')
% % %     ylabel('Test Error (Second Order) ','fontsize',20)
% % %     xlabel('SNR(dB)','fontsize',20)
% % %     set(gca,'FontName','Times New Roman','FontSize',20);    
% % %     grid on 
% % %      pause(0.01)
  
    end
end