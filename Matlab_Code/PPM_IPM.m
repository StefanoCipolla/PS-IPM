function [x,y,z,Info] = PPM_IPM(c,A,b,Q,free_variables,...
                             tol,maxit,pc,printlevel,IterStruct,rho,delta,scale)
%  IPM   Primal-dual Regularized interior-point method with decoupled variables.
%
%  This is the driver function of an IPM for solving the
%  quadratic programming problem
%
%    min c'x + 1/2*x'Qx  subject to A*x=b, x_C>=0.       (1)
%
%  printlevel options:
%  0: turn off iteration output
%  1: print primal and dual residual and duality measure
%  2: print centering parameter and step length
%  3: print residuals in the solution of the step equations
%  printlevel Default: 1.
% Authors: S. Cipolla, J. Gondzio.
% ==================================================================================================================== %
% Parameter filling and dimensionality testing.
% -------------------------------------------------------------------------------------------------------------------- %
[m, n] = size(A);
% Make sure that b and c are column vectors of dimension m and n.
if (size(b,2) > 1) b = (b)'; end
if (size(c,2) > 1) c = (c)'; end
if (~isequal(size(c),[n,1]) || ~isequal(size(b),[m,1]) )
    error('problem dimension incorrect');
end

% Make sure that A is sparse and b, c are full.
if (~issparse(A)) A = sparse(A); end
if (~issparse(Q)) Q = sparse(Q); end
if (issparse(b))  b = full(b);   end
if (issparse(c))  c = full(c);   end

% Set default values for missing parameters.
if (nargin < 5 || isempty(free_variables)) free_variables = []; end
if (nargin < 6 || isempty(tol))            tol = 1e-4;          end
if (nargin < 7 || isempty(maxit))          maxit = 100;         end
if (nargin < 8 || isempty(pc))             pc = 1;              end
if (nargin < 9 || isempty(printlevel))     printlevel = 1;      end
pl = printlevel;
% ==================================================================================================================== %
% Initialization of the structure for convergence hystory
Info = struct();
  
% ==================================================================================================================== %
% Initialization - Mehrotra's Initial Point for QP:
% Choose an initial starting point (x,y,z). For that, we ignore the non-negativity constraints, as well as the
% regularization variables and solve the relaxed optimization problem (which has a closed form solution). Then,
% we shift the solution, to respect the non-negativity constraints. The point is expected to be well centered.
% -------------------------------------------------------------------------------------------------------------------- %
A_tr = A';                                  % Store the transpose for computational efficiency.
pos_vars = setdiff((1:n)',free_variables);
num_of_pos_vars = size(pos_vars,1);
e_pos_vars = ones(num_of_pos_vars,1); 

if (num_of_pos_vars == 0 && pc ~= false)    % Turn off Predictor-Corrector when PMM is only running.
    pc = 1;
end

% =================================================================================================================== %
% Use PCG to solve two least-squares problems for efficiency (along with the Jacobi preconditioner). 
% ------------------------------------------------------------------------------------------------------------------- %
D = sum(A.^2,2) + 10;
Jacobi_Prec = @(x) (1./D).*x;
NE_fun = @(x) (A*(A_tr*x) + 10.*x);
x = pcg(NE_fun,b,10^(-8),min(1000,m),Jacobi_Prec);
x = A_tr*x;
y = pcg(NE_fun,A*(c+Q*x),10^(-8),min(1000,m),Jacobi_Prec);
z = c+ Q*x - A_tr*y;
% =================================================================================================================== %
if (norm(x(pos_vars)) <= 10^(-4)) 
    x(pos_vars) = 0.1.*ones(num_of_pos_vars,1); % 0.1 is chosen arbitrarily
end

if (norm(z(pos_vars)) <= 10^(-4))
    z(pos_vars) = 0.1.*ones(num_of_pos_vars,1); % 0.1 is chosen arbitrarily
end

delta_x = max(-1.5*min(x(pos_vars)),0);
delta_z = max(-1.5*min(z(pos_vars)), 0);
temp_product = (x(pos_vars) + (delta_x.*e_pos_vars))'*(z(pos_vars) + (delta_z.*e_pos_vars));
delta_x_bar = delta_x + (0.5*temp_product)/(sum(z(pos_vars),1)+num_of_pos_vars*delta_z);
delta_z_bar = delta_z + (0.5*temp_product)/(sum(x(pos_vars),1)+num_of_pos_vars*delta_x);

z(pos_vars) = z(pos_vars) + delta_z_bar.*e_pos_vars;
x(pos_vars) = x(pos_vars) + delta_x_bar.*e_pos_vars;
z(free_variables) = 0;

if (issparse(x))  x = full(x); end
if (issparse(z))  z = full(z); end
if (issparse(y))  y = full(y); end
% ==================================================================================================================== %
% PPM parameters initialization

iter        = 0;
PPM_red     = 0.7;
IPM_maxit   = 50;
IPM_Tot_It = 0;

xk = x;
yk = y;
zk = z;

% ********************************************************************
% PPM-IPM Main-Loop
% =============
%r = natural_PPM_res(c,A,b,Q,xk,yk,pos_vars);
[nr_res_p,nr_res_d,mu] = IPM_Res(c,A,A_tr,b,Q,x,y,z,pos_vars,num_of_pos_vars); 
tic 
while (iter < maxit)     
    
    iter = iter+1;
    
    
    % Check for termination. We have found a sufficiently accurate
    % solution if the natural residual is below tol.
    % Info.NatRes(iter) = r;
    Info.NatRes(iter).primal = norm(nr_res_p,1);
    Info.NatRes(iter).dual   = norm(nr_res_d,'inf');
    Info.NatRes(iter).compl  = mu;
    
    if printlevel>0
      fprintf('==================\n');  
      fprintf('PPM iteration: %4d\n', iter);
%       fprintf('Natural Residual: %8.2e\n', Info.NatRes(iter));
      fprintf('NR Primal: %8.2e, NR Dual: %8.2e, NR Comp: %8.2e\n',...
          Info.NatRes(iter).primal,Info.NatRes(iter).dual,Info.NatRes(iter).compl);
      fprintf('==================\n');
    end
    
    STOP = min([abs(x(pos_vars).*z((pos_vars))),abs(z(pos_vars)),abs(x(pos_vars))],[],2);

    STOP = max(STOP);

    if isempty(STOP)
        STOP = 0;
    end

    if ( norm(nr_res_p,1) < tol*scale && norm(nr_res_d,'inf') < tol*scale &&  STOP < tol  )
      if printlevel > 0
      fprintf('optimal solution found\n');
      end
      Info.opt    = 1;
      Info.ExIt   = iter-1;
      Info.IPM_It = IPM_Tot_It;
      break;
    end
    
    
    [x,y,z,Info.IPM(iter)] = prox_eval(c,A,A_tr,Q,b,xk,yk,zk,rho,delta,...
                                       free_variables,pos_vars,num_of_pos_vars,...
                                       PPM_red^iter,IPM_maxit,pc,printlevel);
    
    IPM_Tot_It = Info.IPM(iter).IPMIter+IPM_Tot_It;                               
    
    if Info.IPM(iter).opt == 2 
       Info.ExIt   = iter; 
       Info.opt    = 2;
       Info.IPM_It = IPM_Tot_It;
       break;
    end
    
    
    %r = natural_PPM_res(c,A,b,Q,x,y,pos_vars);
    [nr_res_p,nr_res_d,mu] = IPM_Res(c,A,A_tr,b,Q,x,y,z,pos_vars,num_of_pos_vars); 
    
    
    
     xk = x;
     yk = y;
     zk = z;
  
    if  iter == maxit
        Info.opt     = 0;
        Info.ExIt    = iter;
        Info.IPM_It = IPM_Tot_It;
    end 
    
    
    
    % The PPM has terminated either because the solution accuracy is
    % reached or the maximum number of iterations is exceeded. Print
    % result.
    
        
end

IPMTT = toc;

    if  printlevel>0
        fprintf('time: %g\n', IPMTT);
    end
end
