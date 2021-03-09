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
Ns_NOMA=Ns;
 
snr_value=20;

P=mt;
nVar= sqrt(P/mr/10^(snr_value/10));

nb=round(N/20); %500; %mini batch of data

MaxNumgrad=300*Ns*N;

% theta=theta_part;

interval=130; 
max_real=1; 

index=randperm(size(thetaLocal_set,2));
thetaLocal_set=thetaLocal_set(:,index,:); 

for ave_index=1:interval
    
    ave_index
    
    thetaLocal=thetaLocal_set(:,ave_index:interval:Ns*interval,:);
    thetaLocal_NOMA=thetaLocal_set(:,ave_index:interval:Ns_NOMA*interval,:);
    
    for channel_real=1:max_real 
    
        for k=1:K
            alpha(k)=P*Ns/sum(sum(thetaLocal(:,:,k).^2))/2;  
            alpha_NOMA(k)=P*Ns_NOMA/sum(sum(thetaLocal_NOMA(:,:,k).^2))/2; 
            E(:,:,k)= [eye(d);eye(d)];
            z_base(:,:,k)=randn(mr,Ns);
        end

        z_base_NOMA=randn(mr,Ns_NOMA);


        z=nVar*z_base;
        z_NOMA=nVar* z_base_NOMA;

       % receive signal
         recv_theta_NOMA=0;

       for k=1:K
            recv_theta(:,:,k)= sqrt(alpha(k))*E(:,:,k)*thetaLocal(:,:,k)+z(:,:,k) ; 
            recv_theta_NOMA=recv_theta_NOMA+E(:,:,k)*thetaLocal_NOMA(:,:,k);    
       end
       recv_theta_NOMA=  ( sqrt(min(alpha_NOMA))*recv_theta_NOMA+ z_NOMA );


        grad_index=1; 

        % outer iteration times 
        T=MaxNumgrad/Ns/N; %total intertaion=T*nb

        T_NOMA=MaxNumgrad/Ns_NOMA/N;



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    %%%%% Gaussian Solution %%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            est_theta_gauss=0;
            est_theta_ana=0;
            norm_gauss=0;
            norm_Qk=0;
            
            clear D_gauss

            for k=1:K


                mean_k=mean( 1/sqrt(alpha(k))* pinv(E(:,:,k))*recv_theta(:,:,k),2);
                cov_k=( 1/sqrt(alpha(k))* pinv(E(:,:,k))*recv_theta(:,:,k)- mean_k)*( 1/sqrt(alpha(k))* pinv(E(:,:,k))*recv_theta(:,:,k)- mean_k)'/(Ns-1);

                [u s v]=svd(cov_k);
                s=diag(s);
                s(1:min(Ns-1,d))=1./s(1:min(Ns-1,d));

                D_gauss(:,:,k)= u*diag(s)*v';

                est_theta_gauss=est_theta_gauss+D_gauss(:,:,k)*1/sqrt(alpha(k))* pinv(E(:,:,k))*recv_theta(:,:,k);
                norm_gauss=norm_gauss+D_gauss(:,:,k); 


            end


            clear D_gauss_recv 
            est_theta_gauss=inv(norm_gauss)* est_theta_gauss; 
            for k=1:K

                D_gauss_recv(:,:,k)=inv(norm_gauss)* D_gauss(:,:,k)*1/sqrt(alpha(k))* pinv(E(:,:,k));           
            end

           Err_cov_GaussOMA((ave_index-1)*max_real+channel_real)= sum(sum(abs(est_theta_gauss*est_theta_gauss'/Ns-Cov_global)./abs(Cov_global)))/d/d;
%            Err_mean_GaussOMA= sum( abs(mean(est_theta_gauss,2)-Mean_global)./abs(Mean_global) )/d;

           for k=1:K

                recv_sample_loc=pinv(sqrt(alpha(k))*E(:,:,k))*(sqrt(alpha(k))*E(:,:,k)*thetaLocal(:,:,k)+z(:,:,k)) ; 

                errCov_signle_worker((ave_index-1)*max_real+channel_real,k)=sum(sum(abs(recv_sample_loc*recv_sample_loc'/Ns-Cov_global)./abs(Cov_global)))/d/d;
%                 errMean_signle_worker(k)=sum(abs(mean(recv_sample_loc,2)-Mean_global)./abs(Mean_global))/d;
            end
       
       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Gradient Descent OMA %%%%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         
            %initialize        
            clear  LB_ELBO  Err_cov_GD D_OMA conv_mark_optD

            D_OMA=D_gauss_recv;
            est_theta_GD=est_theta_gauss;

            %learning rate
            etaD= 1e-6; % 1e-6; for GD     1e-8 for nb=500


            tic
            OptD_OMA
            toc

            Err_cov_GD_OMA((ave_index-1)*max_real+channel_real,:)=Err_cov_GD;
    %         ELBO_GD_OMA(ave_index,:)=-LB_ELBO;  % value of min L 




        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Gradient Descent NOMA %%%%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
       
            clear  Err_cov_GD LB_ELBO D_NOMA  conv_mark_optD 

            D_NOMA=eye(d)/K* pinv(sqrt(min(alpha_NOMA))*E(:,:,1));
            est_theta_GD=D_NOMA*recv_theta_NOMA;
            %  learning rate
            etaD=2*1e-7;        % for nb=500 1e-9; for GD 1e-7

            tic
            OptD_NOMA
            toc
 
   
        Err_cov_GD_NOMA((ave_index-1)*max_real+channel_real,:)=Err_cov_GD;
 
        clear recv_theta Err_mean_GD Err_cov_GD LB_ELBO D_OMA
               
    end 
    
    plot([1:T]*Ns*N, mean(Err_cov_GD_OMA,1))
    hold on 
    plot([1:T_NOMA]*Ns_NOMA*N, mean(Err_cov_GD_NOMA,1))
    hold off
    legend('OMA','NOMA')
    pause(0.01)
%         tic
%         Cenrl_SGLD
%         toc
        

end