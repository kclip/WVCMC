clear all
clc
close all


rng(0)




%%% Load Global Sample 
load('~Loacation\BayesianProbit_SyntheticData\Synthetic Data\Global_Sample_SyntheticData.mat')
Cov_global=theta(:, burin+1:size(theta,2))*theta(:,burin+1:size(theta,2))'/length([burin+1:size(theta,2)]);
Mean_global=mean(theta(:, burin+1:size(theta,2)),2); 
clear theta  


N=size(X,1);
% X=[ones(N,1),X];
d=size(X,2);
 
 
mt=2*d+2;
mr=2*d;

 
Comm_Round=1000; 
interval=10; 


max_real=40/interval; 
 
snr_value=20;
P=mr;   % rx poower 
nVar= sqrt(P/mr/10^(snr_value/10));

skewness_set=[0:0.5:3]; 

K=10; 

Ns=Comm_Round/K; 
Ns_NOMA= Comm_Round;   %comm blocks=1000

for shuffle_time=1:max_real
    
    skew_index=1;
    z_set=nVar*randn(mr,Comm_Round*interval);
    
    for sk=skewness_set
        
        for k=1:K         
            path_name=['~Loacation\BayesianProbit_SyntheticData\Synthetic Data\Heterogeneity\SyntheticData_Het=',num2str(sk),'_LocalSample_K10_k=',num2str(k),'.mat'];
            load(path_name)
            thetaLocal_set(:,:,k)=thetaLocal_set_k;
            clear theta  
        end
        
        index=randperm(size(thetaLocal_set,2));
        thetaLocal_set=thetaLocal_set(:,index,:); 

        
        clear thetaLocal thetaLocal_NOMA 
        
        for split_index=1:interval
            
            thetaLocal=thetaLocal_set(:,split_index:interval:Ns*interval,:);
            thetaLocal_NOMA=thetaLocal_set(:,split_index:interval:Ns_NOMA*interval,:);

            z_NOMA=z_set(:,Ns_NOMA*(split_index-1)+1:Ns_NOMA*split_index);
            z=reshape(z_NOMA, mr,Comm_Round/K, K);
            
            
            recv_theta_NOMA=0;

            clear alpha alpha_NOMA E recv_theta
            for k=1:K
                alpha(k)=P*Ns/sum(sum(thetaLocal(:,:,k).^2))/2;  
                alpha_NOMA(k)=P*Ns_NOMA/sum(sum(thetaLocal_NOMA(:,:,k).^2))/2; 
                E(:,:,k)= [eye(d);eye(d)];
                
                recv_theta(:,:,k)= sqrt(alpha(k))*E(:,:,k)*thetaLocal(:,:,k)+z(:,:,k) ; 
                recv_theta_NOMA=recv_theta_NOMA+E(:,:,k)*thetaLocal_NOMA(:,:,k);    
            end
            recv_theta_NOMA=  ( sqrt(min(alpha_NOMA))*recv_theta_NOMA+ z_NOMA );
            
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        %%%%% Gaussian Solution %%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        est_theta_gauss=0;
        est_theta_ana=0;
        norm_gauss=0;
        norm_Qk=0;
        
        clear D_gauss  D_ana 

        for k=1:K
            
            mean_k=mean( 1/sqrt(alpha(k))* pinv(E(:,:,k))*recv_theta(:,:,k),2);
            cov_k=( 1/sqrt(alpha(k))* pinv(E(:,:,k))*recv_theta(:,:,k)- mean_k)*( 1/sqrt(alpha(k))* pinv(E(:,:,k))*recv_theta(:,:,k)- mean_k)'/(Ns-1);

%             [u s v]=svd(cov_k);
%             s=diag(s);
%             s(1:min(Ns-1,d))=1./s(1:min(Ns-1,d));

            D_gauss(:,:,k)= inv(cov_k);   %u*diag(s)*v';

            est_theta_gauss=est_theta_gauss+D_gauss(:,:,k)*1/sqrt(alpha(k))* pinv(E(:,:,k))*recv_theta(:,:,k);
            norm_gauss=norm_gauss+D_gauss(:,:,k); 

                   
          
            
            Qk=cov_k-nVar^2/alpha(k)*pinv(E(:,:,k))*pinv(E(:,:,k))';
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
            est_theta_ana=est_theta_ana+D_ana(:,:,k)*1/sqrt(alpha(k))* pinv(E(:,:,k))*recv_theta(:,:,k);
            
        end
        
        
        
            
       est_theta_gauss=inv(norm_gauss)*est_theta_gauss;
       [un sn vn]=eig(norm_Qk);
       est_theta_ana= inv(norm_Qk)*est_theta_ana;    %un*(inv(sn))*vn'*est_theta_ana;
       entropy_term=0;
       entropy_term_ana=0;
       
       clear D_gauss_recv  D_ana_recv 
        for k=1:K
            D_gauss_recv(:,:,k)=inv(norm_gauss)* D_gauss(:,:,k)*1/sqrt(alpha(k))* pinv(E(:,:,k));
            D_ana_recv(:,:,k)=inv(norm_Qk)*D_ana(:,:,k)*1/sqrt(alpha(k))* pinv(E(:,:,k));
            
            theta_single=1/sqrt(alpha(k))* pinv(E(:,:,k))*recv_theta(:,:,k);
            Err_cov_SingleWorker((shuffle_time-1)*interval+split_index,skew_index,k)= sum(sum(abs(theta_single*theta_single'/Ns-Cov_global)./ abs(Cov_global)))/d/d;
        end

       Err_cov_GaussOMA((shuffle_time-1)*interval+split_index,skew_index) = sum(sum(abs(est_theta_gauss*est_theta_gauss'/Ns-Cov_global)./abs(Cov_global)))/d/d;
       Err_cov_ana((shuffle_time-1)*interval+split_index,skew_index) = sum(sum(abs(est_theta_ana*est_theta_ana'/Ns-Cov_global)./abs(Cov_global)))/d/d;
       


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Gradient Descent OMA %%%%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         
        clear  LB_ELBO  Err_cov_GD D_OMA conv_mark_optD

        D_OMA=D_gauss_recv;
        est_theta_GD=est_theta_gauss;
        
       
        T=50;

        %learning rate
        etaD= 5*1e-6; 
        % 1e-5 for K=5; 
        % 1e-6 for K=20


        nb=N;
        tic
        OptD_OMA
        toc

        Err_cov_GD_OMA((shuffle_time-1)*interval+split_index,skew_index)=Err_cov_GD(t);
  
    
       
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% WGCMC NOMA %%%%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

            mean_recv=mean(pinv(sqrt(min(alpha_NOMA))*E(:,:,1))*recv_theta_NOMA,2);
            cov_recv=(pinv(sqrt(min(alpha_NOMA))*E(:,:,1))*recv_theta_NOMA- mean_recv)*(pinv(sqrt(min(alpha_NOMA))*E(:,:,1))*recv_theta_NOMA- mean_recv)'/(Ns_NOMA-1);
           
            C0=(cov_recv-nVar^2*pinv(sqrt(min(alpha_NOMA))*E(:,:,1))*pinv(sqrt(min(alpha_NOMA))*E(:,:,1))')/K; 
            [u0 s0 v0]=svd(C0);      
            s0(find(s0<0))=0;
            C0_sqrt=u0*sqrt(s0)*v0';
            
            [u s v]=svd(cov_recv);
            inv_cov_sqrt=u*sqrt(inv(s))*v';
           
            
            W_gcmcN=1/sqrt(K)*C0_sqrt*inv_cov_sqrt; 
            est_theta_gcmcN= W_gcmcN*pinv(sqrt(min(alpha_NOMA))*E(:,:,1))*recv_theta_NOMA;
            Err_cov_wgcmcN((shuffle_time-1)*interval+split_index,skew_index) = sum(sum(abs(est_theta_gcmcN*est_theta_gcmcN'/Ns_NOMA-Cov_global)./abs(Cov_global)))/d/d;
            
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Gradient Descent NOMA %%%%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
       
        clear  Err_cov_GD LB_ELBO D_NOMA  conv_mark_optD 


        T_NOMA=50;
        
        nb=N; 


        D_NOMA=eye(d)/K* pinv(sqrt(min(alpha_NOMA))*E(:,:,1));
        est_theta_GD=D_NOMA*recv_theta_NOMA;
        %  learning rate
        etaD=5*1e-7;        % for nb=500 1e-9; for GD 1e-7
        
        % 1e-7 for K=20
        
        
        tic
        OptD_NOMA
        toc
 
        Err_cov_GD_NOMA((shuffle_time-1)*interval+split_index,skew_index)=Err_cov_GD(t);

            

        end
        skew_index=skew_index+1

    end
    
    (shuffle_time-1)*interval+split_index
    
  
    figure(1)
    err_single=reshape(mean(Err_cov_SingleWorker,1),length(skewness_set),K); 
    semilogy(skewness_set, min(err_single'),'-.','LineWidth',2,'Color',[0.4940    0.1840    0.5560])
    hold on 
    semilogy(skewness_set,mean(Err_cov_GaussOMA,1),':','LineWidth',2,'Color',[0.9290    0.6940    0.1250])
    hold on 
    semilogy(skewness_set,mean(Err_cov_ana,1),'--d','LineWidth',2,'Color',[ 0.8500    0.3250    0.0980]);
    hold on 
    semilogy(skewness_set,mean(Err_cov_wgcmcN,1),'--*','LineWidth',2,'Color',[ 0    0.4470    0.7410]);
    hold on
    semilogy(skewness_set,mean(Err_cov_GD_OMA,1),'-d','LineWidth',2,'Color',[ 0.8500    0.3250    0.0980]);
    hold on 
    semilogy(skewness_set,mean(Err_cov_GD_NOMA,1),'-*','LineWidth',2,'Color',[ 0    0.4470    0.7410]);   
    hold off
    legend('Best Single Worker','GCMC','WGCMC (OMA)','WGCMC (NOMA)','WVCMC (OMA)','WVCMC (NOMA)')
    ylabel('Test Error (Second Order) ','fontsize',20)
    xlabel('Skewness','fontsize',20)
    set(gca,'FontName','Times New Roman','FontSize',20);    
    grid on 
     pause(0.01)
    
  

end

