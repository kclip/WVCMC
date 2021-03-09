     for t=1:T
            
          
             
            index=randperm(size(X,1));
            X=X(index,:);
            Y=Y(index); 
            
              
            for m=1:floor(size(X,1)/nb)
                
                if m<floor(size(X,1)/nb) 
                    X_batch= X((m-1)*nb+1:m*nb,:); 
                    Y_batch=Y((m-1)*nb+1:m*nb); 
                else
                    X_batch= X((m-1)*nb+1:size(X,1),:); 
                    Y_batch=Y((m-1)*nb+1:size(X,1)); 
                end
                    
                mid=est_theta_GD'*X_batch';
                rev_normcdf=normcdf(mid);
                rev_normcdf(find(rev_normcdf==1))=1-10^-10;
                rev_normcdf(find(rev_normcdf==0))=0+10^-10;

                Grad_theta = -est_theta_GD/(sigma_prior^2)+ N/nb* (normpdf(mid).* (Y_batch'- normcdf(mid))./ (rev_normcdf.*(1-rev_normcdf)) *X_batch)';

   
   
        
            
            est_theta_GD=0;
           

            for k=1:K
                gradD=1/Ns*Grad_theta*( recv_theta(:,:,k) )'+1/2/K*( inv( D_OMA(:,:,k)*sqrt(alpha(k))*E(:,:,k) )' *(sqrt(alpha(k))*E(:,:,k))' + pinv( D_OMA(:,:,k) )' );            
                conv_mark_optD(k,t)=norm(gradD,'fro');
                D_OMA(:,:,k)=D_OMA(:,:,k)+ etaD*gradD;
                
                est_theta_GD=est_theta_GD+D_OMA(:,:,k)* recv_theta(:,:,k) ;

            end
            
     


            end
            
                        
            
%             Err_mean_GD(t)=sum(abs(sum(est_theta_GD,2)/Ns-sum(theta_test,2)/N_test)./(abs(sum(theta_test,2)/N_test)))/d;
            Err_cov_GD(t)= sum(sum(abs(est_theta_GD*est_theta_GD'/Ns-Cov_global)./abs(Cov_global)))/d/d;
%             Err_mean_GD(t)= sum( abs(mean(est_theta_GD,2)-Mean_global)./abs(Mean_global) )/d;
%             sum(diag(abs((est_theta_GD-mean(est_theta_GD,2))*(est_theta_GD-mean(est_theta_GD,2))'/Ns-Cov_global))./diag(abs(Cov_global)))/d;
        
            entropy_term=0;
            for k=1:K
                entropy_term= entropy_term+ log(abs(det(K*D_OMA(:,:,k)*sqrt(alpha(k))*E(:,:,k))))+  0.5*log(det(d*D_OMA(:,:,k)*D_OMA(:,:,k)'));
            end
            

             mid=est_theta_GD'*X';
            rev_normcdf=normcdf(mid);
            rev_normcdf(find(rev_normcdf==1))=1-10^-10;
            rev_normcdf(find(rev_normcdf==0))=0+10^-10;
        
            LB_ELBO(t)= -1/2/sigma_prior^2*sum(sum(est_theta_GD.^2))/Ns+ sum( Y'*log(rev_normcdf)'+(1-Y')*log(1-rev_normcdf)')/Ns    +entropy_term/2/K;
            

            
            
%             if mod(t,10)==0
%                 figure(1)
%                 plot(LB_ELBO)
%                 pause(0.1) 
%                 figure(2)
%                 plot(Err_cov_GD)
%                 pause(0.1)
% %                 figure(3)
% %                 plot(Err_mean_GD)
% %                 pause(0.1)
%             end
                
     end