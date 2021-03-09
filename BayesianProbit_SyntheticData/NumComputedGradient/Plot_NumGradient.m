
    clear all
    clc
    close all
    
    load('WVCMC_NumGradient_Result.mat')
    load('SGLD_NumGradient_Result.mat')
    plot([1:T]*Ns*N,mean(Err_cov_GD_OMA,1),'-.','LineWidth',2,'Color',[ 0.8500    0.3250    0.0980])   
    hold on 
    plot([1:T_NOMA]*Ns_NOMA*N,mean(Err_cov_GD_NOMA,1),'LineWidth',2,'Color',[ 0    0.4470    0.7410])   
    hold on    
    plot([(t_mix+Ns):MaxNumgrad/nb]*nb,mean(Err_cov_SGLD_all(:,(t_mix+Ns):MaxNumgrad/nb),1),'--','LineWidth',2,'Color',[ 0.4660    0.6740    0.1880])   
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
    
      
    tt_sgld=sort(Err_cov_SGLD_all(:,(t_mix+Ns):MaxNumgrad/nb),'descend');
    tt_sgldmax=tt_sgld(6,:);
    tt_sgldmin=tt_sgld(95,:);
    fill([  [(t_mix+Ns):MaxNumgrad/nb]*nb ,fliplr( [(t_mix+Ns):MaxNumgrad/nb]*nb  )]',[(tt_sgldmax),fliplr(tt_sgldmin)]', 1,'facecolor', [ 0.4660    0.6740    0.1880], 'edgecolor', 'none', 'facealpha', 0.2);
    hold off
    
    legend('WVCMC (OMA)','WVCMC (NOMA)','SGLD')
    ylabel('Test Error (Second Order) ','fontsize',20)
    xlabel('Number of Computed Gradients','fontsize',20)
    set(gca,'FontName','Times New Roman','FontSize',20);    
    grid on 
    ylim([0,0.08])
    xlim([0,2.14*10^7])
    
   figure(2) 
    plot([1:T],mean(Err_cov_GD_OMA,1),'-.','LineWidth',2,'Color',[ 0.8500    0.3250    0.0980])   
    hold on 
    plot([1:T_NOMA],mean(Err_cov_GD_NOMA,1),'LineWidth',2,'Color',[ 0    0.4470    0.7410])   
    hold on 
    xlim([1,120])
    xticks([0, 10, 30, 50])
       ylabel('Test Error (Second Order) ','fontsize',20)
    xlabel('Iterations','fontsize',20)
    set(gca,'FontName','Times New Roman','FontSize',20);    
    grid on 
    
   
    figure(3) 
    plot([1:T],mean(Err_cov_GD_OMA,1),'-.','LineWidth',2,'Color',[ 0.8500    0.3250    0.0980])   
    hold on 
    plot([1:T_NOMA],mean(Err_cov_GD_NOMA,1),'LineWidth',2,'Color',[ 0    0.4470    0.7410])   
    hold on 

    plot([(t_mix+Ns):MaxNumgrad/nb],mean(Err_cov_SGLD_all(:,(t_mix+Ns):MaxNumgrad/nb),1),'--','LineWidth',2,'Color',[ 0.4660    0.6740    0.1880])   
    hold on 
    legend('WVCMC (OMA)','WVCMC (NOMA)','SGLD')
    ylabel('Test Error (Second Order) ','fontsize',20)
    xlabel('Iterations','fontsize',20)
    set(gca,'FontName','Times New Roman','FontSize',20);    
    grid on 
    xlim([0,7*10^4])
    