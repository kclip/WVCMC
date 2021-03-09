function theta = GibbsSampingFunc(X,Y,sigma_prior,Ns,interval, burin,k); %theta column vector
    
  N=size(X,1);
  d=size(X,2);
  theta0=mvnrnd(zeros(d,1), sigma_prior^2*eye(d))';

  C=inv( (X'*X)+eye(d)/sigma_prior^2 );
  
  for t=1:3000
   mid=theta0'*X';
   rev_normcdf=normcdf(mid);
   rev_normcdf(find(rev_normcdf==1))=1-10^-10;
   rev_normcdf(find(rev_normcdf==0))=0+10^-10;
   
   grad = -theta0/(sigma_prior^2)+ (normpdf(mid).* (Y'- normcdf(mid))./ (rev_normcdf.*(1-rev_normcdf)) *X)';
   
%    recod_conv(t)=norm(grad); 
   
   theta0=theta0+0.0001*grad;   
%    acc(t)=sum(abs((sign(X*theta0)+1)/2-Y))/size(Y,1);
  end
  
%   figure(1)
% semilogy(recod_conv)
% figure(2)
% plot(acc) 



  NumIter=Ns*interval+burin; 
 
  theta=zeros(d,NumIter);
  theta=[theta0, theta];
  
  
  for s=1: NumIter

    nu=zeros(N,1);
    for i=1:N
        pd = makedist('Normal',theta(:,s)'*X(i,:)',1);     % Normal(mu,sigma)
        if Y(i)>0
            pdt = truncate(pd,0,inf);              % truncated to interval (0,1)
        else
            pdt = truncate(pd,-inf,0);
        end
        
        nu(i) = random(pdt,1,1);
    end
    
    theta(:,s+1)=mvnrnd(C*X'*nu,(C+C')/2,1)';

  end
   
  theta=theta(:,burin+2:NumIter+1);
  
  data_name=['MNIST01_Dim30_LocalSample_K10_k=',num2str(k),'.mat'];

  save(data_name,'theta','-mat');
   
end