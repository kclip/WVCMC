  
function Single_Expriment_SNR(X,Y,thetaLocal,thetaLocal_NOMA, shuffle_time, interval,split_index,K,d,mr,snr_set,P,Ns,Ns_NOMA,Cov_global,N,sigma_prior); %theta column vector
    

    snr_index=1; 
         
    for k=1:K
        alpha(k)=P*Ns/sum(sum(thetaLocal(:,:,k).^2))/2;  
        alpha_NOMA(k)=P*Ns_NOMA/sum(sum(thetaLocal_NOMA(:,:,k).^2))/2; 
        E(:,:,k)= [eye(d);eye(d)];
        z_base(:,:,k)=randn(mr,Ns);
    end


    z_base_NOMA=randn(mr,Ns_NOMA);
    
    
 

    for snr_value=snr_set
        
        
     
        nVar= sqrt(P/mr/10^(snr_value/10));
        z=nVar*z_base;
        z_NOMA=nVar* z_base_NOMA;
        
        
        % receive signal
        recv_theta_NOMA=0;

        for k=1:K
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
        D_gauss=zeros(d,d,K); 

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
       
        for k=1:K
            D_gauss_recv(:,:,k)=inv(norm_gauss)* D_gauss(:,:,k)*1/sqrt(alpha(k))* pinv(E(:,:,k));
            D_ana_recv(:,:,k)=inv(norm_Qk)*D_ana(:,:,k)*1/sqrt(alpha(k))* pinv(E(:,:,k));
            
            theta_single=1/sqrt(alpha(k))* pinv(E(:,:,k))*recv_theta(:,:,k);
            Err_cov_SingleWorker(snr_index,k)= sum(sum(abs(theta_single*theta_single'/Ns-Cov_global)./ abs(Cov_global)))/d/d;
        end

       Err_cov_GaussOMA(snr_index) = sum(sum(abs(est_theta_gauss*est_theta_gauss'/Ns-Cov_global)./abs(Cov_global)))/d/d;
       Err_cov_ana(snr_index) = sum(sum(abs(est_theta_ana*est_theta_ana'/Ns-Cov_global)./abs(Cov_global)))/d/d;
       


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Gradient Descent OMA %%%%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         
        clear  LB_ELBO  Err_cov_GD D_OMA conv_mark_optD

        D_OMA=D_gauss_recv;
        est_theta_GD=est_theta_gauss;
        
        T=250; 
%         if snr_value<20
%             T=1000;
%         else 
%             T=50;
%         end
        %learning rate
        etaD= 1e-6; % 1e-6 for 30dB


        nb=round(N/20);
%         tic
        OptD_OMA
%         toc

        Err_cov_GD_OMA(snr_index)=Err_cov_GD(t);
  
    
       
        
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
            Err_cov_wgcmcN(snr_index) = sum(sum(abs(est_theta_gcmcN*est_theta_gcmcN'/Ns_NOMA-Cov_global)./abs(Cov_global)))/d/d;
            
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Gradient Descent NOMA %%%%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
       
        clear  Err_cov_GD LB_ELBO D_NOMA  conv_mark_optD 


        T_NOMA=250;
        
        nb=round(N/20); 


        D_NOMA=eye(d)/K* pinv(sqrt(min(alpha_NOMA))*E(:,:,1));
        est_theta_GD=D_NOMA*recv_theta_NOMA;
        %  learning rate
        etaD=2*1e-7;        % for nb=500 1e-9; for GD 1e-7

%         tic
        OptD_NOMA
%         toc
 
        Err_cov_GD_NOMA(snr_index)=Err_cov_GD(t);

               

        snr_index=snr_index+1;
                
        

  
    end
    
    data_name=['MNIST01_Dim30_SNR_index=',num2str((shuffle_time-1)*interval+split_index),'.mat'];

    save(data_name,'Err_cov_wgcmcN','-mat',...
                    'Err_cov_GD_NOMA','-mat',...
                    'Err_cov_GD_OMA','-mat',...
                    'Err_cov_SingleWorker','-mat',...
                    'Err_cov_GaussOMA','-mat',...
                    'Err_cov_ana','-mat');
    
    
end
    