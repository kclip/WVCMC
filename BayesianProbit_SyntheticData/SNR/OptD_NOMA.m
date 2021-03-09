        for t=1:T_NOMA

            

            
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

                gradD=1/Ns_NOMA*Grad_theta*recv_theta_NOMA'+  +1/(K+1)*( K*inv( D_NOMA*(sqrt(min(alpha_NOMA))*E(:,:,1)) )' *(sqrt(min(alpha_NOMA))*E(:,:,1))' + pinv( D_NOMA )' );      
                conv_mark_optD(t)=norm(gradD,'fro');
                D_NOMA=D_NOMA+ etaD*gradD;

                est_theta_GD=D_NOMA* recv_theta_NOMA;
            end
           
         
            Err_cov_GD(t)=   sum(sum(abs(est_theta_GD*est_theta_GD'/Ns_NOMA-Cov_global)./abs(Cov_global)))/d/d;
         
            mid=est_theta_GD'*X';
            rev_normcdf=normcdf(mid);
            rev_normcdf(find(rev_normcdf==1))=1-10^-10;
            rev_normcdf(find(rev_normcdf==0))=0+10^-10;
            
            LB_ELBO(t)= -1/2/sigma_prior^2*sum(sum(est_theta_GD.^2))/Ns_NOMA+ sum( Y'*log(rev_normcdf)'+(1-Y')*log(1-rev_normcdf)')/Ns_NOMA  ...
                         + (K*log(abs(det(d*D_NOMA*E(:,:,1))))+0.5*log(det(K*D_NOMA*D_NOMA')) )/(K+1);
                     
                     
                      
% % %             if mod(t,10)==0
% % %                 figure(1)
% % %                 plot(LB_ELBO)
% % %                 pause(0.1) 
% % %                 figure(2)
% % %                 plot(Err_cov_GD)
% % %                 pause(0.1)
% % %                 figure(3)
% % %                 plot(mean(conv_mark_optD,1))
% % %                 pause(0.1)
% % %             end
% % %                 
            

        end