        for t=1:T_NOMA
           
            Grad_theta=-inv(Cov_global)*est_theta_GD;
          
            
            est_theta_GD=0;                
                
            gradD=1/Ns_NOMA*Grad_theta*recv_theta'+   inv(D_NOMA)';            
%             conv_mark_optD(t)=norm(gradD,'fro');
            D_NOMA=D_NOMA+ etaD*gradD;

            est_theta_GD=D_NOMA* recv_theta;
           
         
            Err_cov_GD(t)= sum(sum(abs(est_theta_GD*est_theta_GD'/Ns_NOMA-Cov_global)./abs(Cov_global)))/d/d;
         
       
            LB_ELBO(t)= -0.5*trace(est_theta_GD*est_theta_GD'*inv(Cov_global))/Ns_NOMA +  log(abs(det(D_NOMA))) ;
            
        
        end