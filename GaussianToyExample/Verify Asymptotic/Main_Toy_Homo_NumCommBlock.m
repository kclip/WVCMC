clear all
clc
close all


rng(0)


K=10;

T=300;

T_NOMA=30;

clear X Y

% sigma_prior=2;

d=5;


mt=d;
mr=d;



% Regenerae theta and Y to have a zero mean posterior 

rho=[0:0.1:0.9];
Cov_global=0;

for k=1:K  
%     cov_data(:,:,k)= 0.9*eye(d)+0.1*ones(d,d);    ;  %eye(d)+ 20*randn(d,d);
    
    Cov_local(:,:,k)= toeplitz((rho(k)*ones(1,d)).^[0:d-1]);
    
%     (1-rho(k))*eye(d)+rho(k)*ones(d,d);  %inv(cov_data(:,:,k)*cov_data(:,:,k)'+sigma_prior^-2/K*eye(d));
    mean_local(:,k)=zeros(d,1);
    
    Cov_global=Cov_global+inv(Cov_local(:,:,k));
end

Cov_global=inv(Cov_global);
for k=1:K      
    Cov_local(:,:,k)=K*Cov_global; 
end 

mean_global=zeros(d,1);

Ns_set=[100,500,1000,5000,20000,50000,100000,500000]; %[10,100,500,1000,3000,8000,15000,50000,100000]; %[8,10,25,50,80,150,300,600,1000];

snr_value=5; 

% nVar=sqrt(10^(-1.8));   %-18 dbm noise
% P= 10^(snr_value/10)*mr*nVar^2;

P=mr;
nVar= sqrt(P/mr/10^(snr_value/10));



for ave_index=1:100
    ave_index
 
    
    for k=1:K
        thetaLocal_set(:,:,k)=mvnrnd(mean_local(:,k),Cov_local(:,:,k),max(Ns_set))';
        thetaLocal_NOMA_set(:,:,k)=mvnrnd(mean_local(:,k),Cov_local(:,:,k),max(Ns_set)*K)';
    end
   
    
    % Generate Noise for this experiment
    % Calcuate mean covariance for clean data sampels 
    norm_G=0;
    for k=1:K
        z_base(:,:,k)=randn(mr,max(Ns_set));
        H(:,:,k)=eye(d);
    end
    z_base_NOMA=randn(mr,max(Ns_set)*K);
    
    

    
    Ns_index=1; 
    
    
 

    for Ns=Ns_set
        
        Ns_NOMA=Ns*K;
        
        thetaLocal=thetaLocal_set(:,1:Ns,:);
        thetaLocal_NOMA=thetaLocal_NOMA_set(:,1:Ns_NOMA,:);
%         nVar=0;
        z=nVar*z_base(:,1:Ns,:);
        z_NOMA=nVar* z_base_NOMA(:,1:Ns_NOMA);
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        %%%%% Gaussian Solution %%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        est_theta_gauss=0;
        est_theta_ana=0;
        norm_gauss=0;
        norm_Qk=0;
        D_gauss=zeros(d,d,K); 

        for k=1:K
            alpha(k)=P*Ns/sum(sum(thetaLocal(:,:,k).^2));        
            recv_theta(:,:,k)= ( H(:,:,k)*sqrt(alpha(k))*thetaLocal(:,:,k)+z(:,:,k) );   
                   
            mean_k=mean(recv_theta(:,:,k),2);
            cov_k=(recv_theta(:,:,k)- mean_k)*(recv_theta(:,:,k)- mean_k)'/(Ns-1);
            D_gauss(:,:,k)=inv(cov_k/alpha(k));
            
            
     
                        
            est_theta_gauss=est_theta_gauss+D_gauss(:,:,k)*recv_theta(:,:,k)/sqrt(alpha(k));
            norm_gauss=norm_gauss+D_gauss(:,:,k); 
            
            Qk=(cov_k-nVar^2*eye(d))/(alpha(k));
            [u s v]=eig(Qk);
            inv_s=inv(s);
            inv_s(find(inv_s<0))=0;
%             
%             inv_s=1./diag(s);
%             inv_s(find(inv_s==Inf))=0;
%     
%             sqrt_Qk_inv= u*diag(sqrt(inv_s))*v';
            norm_Qk=norm_Qk+u*inv_s*v';
            [uc sc vc]=eig(cov_k);
            D_ana(:,:,k)= u*sqrt(inv_s)*v'*uc*sqrt(inv(sc))*vc';
            est_theta_ana=est_theta_ana+D_ana(:,:,k)*recv_theta(:,:,k);
            
        end
        
        
        
            
       est_theta_gauss=inv(norm_gauss)*est_theta_gauss;
       [un sn vn]=eig(norm_Qk);
       est_theta_ana=un*(inv(sn))*vn'*est_theta_ana;
       entropy_term=0;
       entropy_term_ana=0;
       
        for k=1:K
            D_gauss(:,:,k)=inv(norm_gauss)*D_gauss(:,:,k);
            D_ana(:,:,k)=un*(inv(sn))*vn'*D_ana(:,:,k);
        end

%        Err_mean_GaussOMA(ave_index,Ns_index)=sum(abs(sum(est_theta_gauss,2)/Ns-sum(theta_test,2)/N_test)./(abs(sum(theta_test,2)/N_test)))/d;
       Err_cov_GaussOMA(ave_index,Ns_index) = sum(sum(abs(est_theta_gauss*est_theta_gauss'/Ns-Cov_global)./abs(Cov_global)))/d/d;
%         sum(diag(abs((est_theta_gauss-mean(est_theta_gauss,2))*(est_theta_gauss-mean(est_theta_gauss,2))'/Ns-Cov_global))./diag(abs(Cov_global)))/d;
       
       
%        ELBO_GaussOMA(ave_index,Ns_index)=-(-1/2/sigma_prior^2*sum(sum(est_theta_gauss.^2))/Ns-1/2*sum(sum((Y-X*est_theta_gauss).^2))/Ns+entropy_term/2/K);

%        Err_mean_ana(ave_index,Ns_index)=sum(abs(sum(est_theta_ana,2)/Ns-sum(theta_test,2)/N_test)./(abs(sum(theta_test,2)/N_test)))/d;
       Err_cov_ana(ave_index,Ns_index) = sum(sum(abs(est_theta_ana*est_theta_ana'/Ns-Cov_global)./abs(Cov_global)))/d/d;
       %sum(diag(abs((est_theta_ana-mean(est_theta_ana,2))*(est_theta_ana-mean(est_theta_ana,2))'/Ns-Cov_global))./diag(abs(Cov_global)))/d;
%        ELBO_ana(ave_index,Ns_index)=-(-1/2/sigma_prior^2*sum(sum(est_theta_ana.^2))/Ns-1/2*sum(sum((Y-X*est_theta_ana).^2))/Ns+entropy_term_ana/2/K);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Gradient Descent OMA %%%%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         
        %initialize        
        clear alpha LB_ELBO Err_mean_GD Err_cov_GD D_OMA
        D_OMA=D_gauss;
        est_theta_GD=0;
        for k=1:k
            alpha(k)=P*Ns/sum(sum(thetaLocal(:,:,k).^2));        
            recv_theta(:,:,k)= ( H(:,:,k)*sqrt(alpha(k))*thetaLocal(:,:,k)+z(:,:,k) );
            est_theta_GD=est_theta_GD+ D_OMA(:,:,k)*recv_theta(:,:,k);  
        end
      
        %learning rate
        etaD= 5*1e-3;     %1e-8 for 50dB %*(10^(50/10)/ 10^(snr_value/10))^(1/5);      % for over 15 dB 1e-8;    
        
        tic
        OptD_OMA_Gauss
        toc
        
%         close all
%     
%         figure(1)
%         hold on 
%         plot(LB_ELBO,'LineWidth',2)
%         hold off
% % 
% %         figure(2)
% %         plot(Err_mean_GD,'LineWidth',2)
%         
%         figure(3)
%         plot(Err_cov_GD,'LineWidth',2)

        
%         Err_mean_GD_OMA(ave_index,Ns_index)=Err_mean_GD(t);
        Err_cov_GD_OMA(ave_index,Ns_index)=Err_cov_GD(t);
        ELBO_GD_OMA(ave_index,Ns_index)=-LB_ELBO(t);  % value of min L 
 
      
        clear recv_theta Err_mean_GD Err_cov_GD LB_ELBO D_NOMA  alpha 
        recv_theta=0;
        for k=1:k
            alpha(k)=P*Ns_NOMA/sum(sum(thetaLocal_NOMA(:,:,k).^2));        
            recv_theta = recv_theta+  H(:,:,k)*thetaLocal_NOMA(:,:,k);    
        end
        
       
        
        
        recv_theta= ( sqrt(min(alpha))*recv_theta+ z_NOMA );
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% WGCMC NOMA %%%%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
            mean_recv=mean(recv_theta,2);
            cov_recv=(recv_theta- mean_recv)*(recv_theta- mean_recv)'/(Ns_NOMA-1);
           
            C0=(cov_recv-nVar^2*eye(d))/K/min(alpha); 
            [u0 s0 v0]=svd(C0);      
            s0(find(s0<0))=0;
            C0_sqrt=u0*sqrt(s0)*v0';
            
            [u s v]=svd(cov_recv);
            inv_cov_sqrt=u*sqrt(inv(s))*v';
           
            
            W_gcmcN=1/sqrt(K)*C0_sqrt*inv_cov_sqrt; 
            est_theta_gcmcN= W_gcmcN*recv_theta;
            Err_cov_wgcmcN(ave_index,Ns_index) = sum(sum(abs(est_theta_gcmcN*est_theta_gcmcN'/Ns_NOMA-Cov_global)./abs(Cov_global)))/d/d;
            
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Gradient Descent NOMA %%%%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        D_NOMA=eye(d)/K;    
        est_theta_GD=D_NOMA*recv_theta;
%         sum(sum(abs(est_theta_GD*est_theta_GD'/Ns_NOMA-Cov_global)./abs(Cov_global)))/d/d
        %  learning rate
        etaD= 1e-3;  %%10^-10 for index 1 2;   

        tic
        OptD_NOMA_Gauss
        toc

%         close all
%         figure(1)
%         plot(LB_ELBO,'LineWidth',2)
% 
% % 
% %         figure(2)
% %         plot(Err_mean_GD,'LineWidth',2)
% 
%         figure(3)
%         plot(Err_cov_GD,'LineWidth',2)


        % Err_mean_NTNI_NOMA(ave_index,het_index)=Err_mean_GD(t);
        Err_cov_GD_NOMA(ave_index,Ns_index)=Err_cov_GD(t);
        ELBO_GD_OMA(ave_index,Ns_index)=-LB_ELBO(t);  % value of min 

        clear recv_theta Err_mean_GD Err_cov_GD LB_ELBO D_OMA
               

        Ns_index=Ns_index+1;
                
        

  
    end
%     
%     close all 
    
% % %     figure(1)   
% % %     semilogy(Ns_set,mean(Err_mean_GaussOMA,1),'--d','LineWidth',2,'Color',[0    0.4470    0.7410])
% % %     hold on 
% % %     semilogy(Ns_set,mean(Err_mean_ana,1),':d','LineWidth',2,'Color',[ 0.9290    0.6940    0.1250]);
% % % %     hold on
% % % %     semilogy(Ns_set,mean(Err_mean_GD_OMA,1),'-o','LineWidth',2,'Color',[0.8500    0.3250    0.0980])   
% % %     hold off
% % %     legend('GCMC','analytical')
% % %     ylabel('Test Error (First Order) ','fontsize',20)
% % %     xlabel('The Number of Samples','fontsize',20)
% % % %     title('First Order','fontsize',17)
% % %     set(gca,'FontName','Times New Roman','FontSize',20);    
% % %    grid on 
% % %     pause(0.01)

    
    
   figure(2)   
    semilogx(Ns_set*K,mean(Err_cov_GaussOMA,1),':','LineWidth',2,'Color',[0.9290    0.6940    0.1250])
    hold on 
    semilogx(Ns_set*K,mean(Err_cov_ana,1),'--d','LineWidth',2,'Color',[ 0.8500    0.3250    0.0980]);
    hold on 
    semilogx(Ns_set*K,mean(Err_cov_wgcmcN,1),'--*','LineWidth',2,'Color',[ 0    0.4470    0.7410]);
    hold on
    semilogx(Ns_set*K,mean(Err_cov_GD_OMA,1),'-d','LineWidth',2,'Color',[ 0.8500    0.3250    0.0980]);
    hold on 
    semilogx(Ns_set*K,mean(Err_cov_GD_NOMA,1),'-*','LineWidth',2,'Color',[ 0    0.4470    0.7410]);   
    hold off
    legend('GCMC','WGCMC(OMA)','WGCMC(NOMA)','WVCMC(OMA)','W-VCMC(NOMA)')
    ylabel('Test Error (Second Order) ','fontsize',20)
    xlabel('Communication Blocks','fontsize',20)
    set(gca,'FontName','Times New Roman','FontSize',20);    
    grid on 
     pause(0.01)
  
% %     
% %     figure(3) 
% %     plot(Ns_set,mean(ELBO_GaussOMA,1),'--d','LineWidth',2,'Color',[0    0.4470    0.7410])
% %     hold on 
% %     semilogy(Ns_set,mean(ELBO_ana,1),':d','LineWidth',2,'Color',[ 0.9290    0.6940    0.1250]);
% %     hold on
% % %     plot(Ns_set,mean(ELBO_GD_OMA,1),'-o','LineWidth',2,'Color',[0.8500    0.3250    0.0980])   
% %     hold off
% %     legend('GCMC','analytical')
% %     ylabel('Test Error (Second Order) ','fontsize',20)
% %     %     legend('W-VMC OMA (GD)','W-VMC OMA (CVX)', 'W-VMC NOMA (GD)','W-VMC NOMA (CVX)')
% %     ylabel('Objective Function ','fontsize',20)
% %     xlabel('The Number of Samples','fontsize',20)
% %     set(gca,'FontName','Times New Roman','FontSize',20);    
% %       grid on 
  
% %     pause(0.01)
end