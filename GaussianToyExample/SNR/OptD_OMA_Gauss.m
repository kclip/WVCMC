        for t=1:T
            Grad_theta=-inv(Cov_global)*est_theta_GD;
            
            est_theta_GD=0;
           

            for k=1:K
                gradD=1/Ns*Grad_theta*( recv_theta(:,:,k) )'+1/K*inv(D_OMA(:,:,k) );  %   ( inv(D_OMA(:,:,k)*H(:,:,k))' *(H(:,:,k))' +( D_OMA(:,:,k)'*inv(D_OMA(:,:,k)* D_OMA(:,:,k)') )' );            
                conv_mark_optD(k,t)=norm(gradD,'fro');
                D_OMA(:,:,k)=D_OMA(:,:,k)+ etaD*gradD;
                
                est_theta_GD=est_theta_GD+D_OMA(:,:,k)* recv_theta(:,:,k) ;

            end
            
     
            
            
%             Err_mean_GD(t)=sum(abs(sum(est_theta_GD,2)/Ns-sum(theta_test,2)/N_test)./(abs(sum(theta_test,2)/N_test)))/d;
            Err_cov_GD(t)= sum(sum(abs(est_theta_GD*est_theta_GD'/Ns-Cov_global)./abs(Cov_global)))/d/d;
%             sum(diag(abs((est_theta_GD-mean(est_theta_GD,2))*(est_theta_GD-mean(est_theta_GD,2))'/Ns-Cov_global))./diag(abs(Cov_global)))/d;
        
            entropy_term=0;
            for k=1:K
                entropy_term= entropy_term+ log(abs(det(D_OMA(:,:,k))))+  0.5*log(det(D_OMA(:,:,k)*D_OMA(:,:,k)'));
            end
            

            
            LB_ELBO(t)=-0.5*trace(est_theta_GD*est_theta_GD'*inv(Cov_global))/Ns  +entropy_term/2/K;
            
  

        end