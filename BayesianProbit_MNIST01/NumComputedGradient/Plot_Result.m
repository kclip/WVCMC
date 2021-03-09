 

clear all
clc

Err_cov_SGLD=[]; 

for ave_index=1:100
    path_name=['/Users/liudongzhu/Desktop/MATLAB/Bayes MC/Paper/BayesianProbit/NumComputedGradient_MNIST01_Dim30/SGLD_Result/MNIST01_Dim30_SGLD_aveindex=',num2str(ave_index),'.mat'];

    load(path_name);
    
    Err_cov_SGLD=[Err_cov_SGLD; Err_cov_SGLD_all];
end

index_sgld=find(Err_cov_SGLD_all>0); 
Err_cov_SGLD=Err_cov_SGLD(:,index_sgld); 

    

load('Result_VCMC_NumGrad_RevOutLier.mat')
plot([1:T]*Ns*N,mean(Err_cov_GD_OMA,1),'-.','LineWidth',2,'Color',[ 0.8500    0.3250    0.0980])   
hold on 
plot([1:T_NOMA]*Ns_NOMA*N,mean(Err_cov_GD_NOMA,1),'LineWidth',2,'Color',[ 0    0.4470    0.7410])   
hold on    
plot(index_sgld*(N/20),mean(Err_cov_SGLD,1),'--','LineWidth',2,'Color',[ 0.4660    0.6740    0.1880])   
hold on 


tt_OMA=sort(Err_cov_GD_OMA,'descend');
tt_OMAmax=tt_OMA(6,:);
tt_OMAmin=tt_OMA(95,:);
fill([  [1:T]*Ns*N,fliplr( [1:T]*Ns*N)]',[(tt_OMAmax),fliplr(tt_OMAmin)]', 1,'facecolor',[0.8500    0.3250    0.0980] , 'edgecolor', 'none', 'facealpha', 0.2);
hold on 

tt_NOMA=sort(Err_cov_GD_NOMA,'descend');
tt_NOMAmax=tt_NOMA(6,:);
tt_NOMAmin=tt_NOMA(95,:);
fill([  [1:T_NOMA]*Ns_NOMA*N,fliplr( [1:T_NOMA]*Ns_NOMA*N )]',[(tt_NOMAmax),fliplr(tt_NOMAmin)]', 1,'facecolor', [ 0    0.4470    0.7410], 'edgecolor', 'none', 'facealpha', 0.2);
hold on 


tt_sgld=sort(Err_cov_SGLD,'descend');
tt_sgldmax=tt_sgld(6,:);
tt_sgldmin=tt_sgld(95,:);
fill([ index_sgld*(N/20) ,fliplr(index_sgld*(N/20))]',[(tt_sgldmax),fliplr(tt_sgldmin)]', 1,'facecolor', [ 0.4660    0.6740    0.1880], 'edgecolor', 'none', 'facealpha', 0.2);
hold off

legend('WVCMC (OMA)','WVCMC (NOMA)','SGLD')
ylabel('Test Error (Second Order) ','fontsize',20)
xlabel('Number of Computed Gradients','fontsize',20)
    set(gca,'FontName','Times New Roman','FontSize',20);    
ylim([0,60])

