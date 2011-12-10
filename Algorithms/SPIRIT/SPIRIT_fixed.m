%Simulation on updating w (works better for data with mean-zero)
%function [W, k, Proj, recon] = SPIRIT(A, lambda, k0)
%fix the number of hidden varialbes and track them over time
%Inputs:
%A: data matrix
%-lambda: forgetting factor between [0,1], 1 means putting equal weight on
%         the past, otherwise the past is exponentially decayed by lambda
%-k0: the number of hidden variables, default=3
%
%Output: 
%-W: eigvector matrix such as 1st eigvector W(:,1), kth eigvector W(:,k)
%-Proj: the lowe-dim projections;
%-recon: the reconstructions of A; 
%Example:
% >> A = sin((1:1000)/20)'*rand(1,10);
% >> plot(A)
% >> A = [sin((1:1000)/10) sin((1:1000)/50)]'*rand(1,10);
% >> plot(A)
% >> [W,Proj,recon] = SPIRIT_fixed(A,1,3);
% >> plot(recon)
%----------------------------------------------------------------
% Copyright: 2006,
%         Spiros Papadimitriou, Jimeng Sun, Christos Faloutsos.
%         All rights reserved.
% Please address questions or comments to: jimeng@cs.cmu.edu
%----------------------------------------------------------------

function [W, Proj, recon] = SPIRIT(A, lambda, k0)

if nargin < 4, k0 = 3; end

n = size(A,2);
totalTime = size(A, 1);
Proj = zeros(totalTime,n); 
recon = zeros(totalTime,n);
%initialize w_i to unit vectors
W = eye(n);
d = 0.01*ones(n, 1);
m = k0; % number of eigencomponents

relErrors = zeros(totalTime, 1);

sumYSq=0;
sumXSq=0;

%incremental update W
for t = 1:totalTime
  % update W for each y_t
  x = A(t,:)';
  for j = 1:m
     [W(:,j), d(j), x] = updateW(x, W(:,j), d(j), lambda);
     Wj = W(:,j);    
  end
  W(:,1:m) = grams(W(:,1:m));
  %compute low-D projection, reconstruction and relative error
  Y = W(:,1:m)' * A(t,:)'; %project to m-dimensional space
  xActual = A(t,:)'; %actual vector of the current time
  xProj = W(:,1:m) * Y; %reconstruction of the current time
  Proj(t,1:m) = Y; 
  recon(t,:) = xProj;
  xOrth = xActual - xProj;
  relErrors(t) = sum(xOrth.^2)/sum(xActual.^2); 
end

% set outputs
W(:,1:m) = grams(W(:,1:m));
W = W(:,1:m);
errs = relErrors;

