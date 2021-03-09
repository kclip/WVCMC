
clear all
clc

Err_cov_ana_set=[]; 
Err_cov_wgcmcN_set=[];
Err_cov_GaussOMA_set=[];
Err_cov_SingleWorker_set=[];
Err_cov_GD_OMA_set=[];
Err_cov_GD_NOMA_set=[];





for ave_index=1:40
    path_name=['/Users/liudongzhu/Desktop/MATLAB/Bayes MC/Paper/BayesianProbit/SNR_MNIST_Dim30/SNR_Result_MNIST/MNIST01_Dim30_SNR_index=',num2str(ave_index),'.mat'];

    load(path_name);
    Err_cov_ana_set=[Err_cov_ana_set;Err_cov_ana];
    Err_cov_wgcmcN_set=[Err_cov_wgcmcN_set;Err_cov_wgcmcN];
    Err_cov_GaussOMA_set=[Err_cov_GaussOMA_set;Err_cov_GaussOMA];
    Err_cov_SingleWorker_set(ave_index,:,:)=Err_cov_SingleWorker;
    Err_cov_GD_OMA_set=[Err_cov_GD_OMA_set;Err_cov_GD_OMA];
    Err_cov_GD_NOMA_set=[Err_cov_GD_NOMA_set;Err_cov_GD_NOMA];
    
    
end

clearvars -except Err_cov_ana_set Err_cov_wgcmcN_set Err_cov_GaussOMA_set ...
Err_cov_SingleWorker_set Err_cov_GD_OMA_set  Err_cov_GD_NOMA_set

snr_set=[5:5:40];
K=10; 
    figure(1)
    err_single=reshape(mean(Err_cov_SingleWorker_set,1),length(snr_set),K); 
    plot(snr_set, min(err_single'),'-.','LineWidth',2,'Color',[0.4940    0.1840    0.5560])
    hold on 
    plot(snr_set,mean(Err_cov_GaussOMA_set,1),':','LineWidth',2,'Color',[0.9290    0.6940    0.1250])
    hold on 
    plot(snr_set,mean(Err_cov_ana_set,1),'--d','LineWidth',2,'Color',[ 0.8500    0.3250    0.0980]);
    hold on 
    plot(snr_set,mean(Err_cov_wgcmcN_set,1),'--*','LineWidth',2,'Color',[ 0    0.4470    0.7410]);
    hold on
    plot(snr_set,mean(Err_cov_GD_OMA_set,1),'-d','LineWidth',2,'Color',[ 0.8500    0.3250    0.0980]);
    hold on 
    plot(snr_set,mean(Err_cov_GD_NOMA_set,1),'-*','LineWidth',2,'Color',[ 0    0.4470    0.7410]);   
    hold off
    legend('Best Single Worker','GCMC','WGCMC (OMA)','WGCMC (NOMA)','WVCMC (OMA)','WVCMC (NOMA)')
    ylabel('Test Error (Second Order) ','fontsize',20)
    xlabel('SNR(dB)','fontsize',20)
    set(gca,'FontName','Times New Roman','FontSize',20);    
    grid on 


