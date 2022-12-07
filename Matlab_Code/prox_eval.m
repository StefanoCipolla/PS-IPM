% S. Cipolla, J. Gondzio.

function [x,y,z,OInfo] = prox_eval(c,A,A_tr,Q,b,xk,yk,zk,rho,delta,...
                                       free_variables,pos_vars,num_of_pos_vars,...
                                       tol,maxit,pc,pl)
% ==================================================================================================================== %
% This function is an Interior Point-Proximal Method of Multipliers, suitable for solving linear and convex quadratic
% programming problems. The method takes as input a problem of the following form:
%
%                                    min   c^T x + (1/2)x^TQx +
%                                    (rho/2)|x-xk|^2 + (delta/2)|y|^2
%                                    s.t.  A x + delta(y-yk) = b,
%                                          x_C >= 0, for i in C \subset {1,...,n},
%                                          x_F free, for i in F = {1,...,n}\C.
%
% INPUT PARAMETERS:
% IP_PMM(c, A, Q, b): find the optimal solution of the problem, with an error tolerance of 10^(-6).
%                     Upon success, the method returns x (primal solution), y (Lagrange multipliers) and
%                     z >= 0 (dual optimal slack variables). If the run was unsuccessful, the method  either returns
%                     a certificate of infeasibility, or terminates after 100 iterations. By default, the method
%                     scales the constraint matrix.
% IP_PMM(c, A, Q, b, free_variables): The last parameter is a matrix of indices, pointing to the free variables of the
%                                     problem. If not given, it is assumed that there are no free variables.
% IP_PMM(c, A, Q, b, free_variables, tol): This way, the user can specify the tolerance to which the problem is solved.
% IP_PMM(c, A, Q, b, free_variables, tol, max_it): This way, the user can also specify the maximum number of iterations.
% IP_PMM(c, A, Q, b, free_variables, tol, maxit, pc): predictor-corrector option.
%     1 = No Predictor-Corrector, 2 Merhotra P-C, 3 Gondzio's Multiple Corrections
% IP_PMM(c, A, Q, b, free_variables, tol, max_it,pc, printlevel): sets the printlevel.
%                                                              0: turn off iteration output
%                                                              1: print primal and dual residual and duality measure
%                                                              2: print centering parameter and step length
% OUTPUT: [x,y,z,opt,iter], where:
%         x: primal solution
%         y: Lagrange multiplier vector
%         z: dual slack variables
%         opt: true if problem was solved to optimality, false if problem not solved or found infeasible.
%         iter: numeber of iterations to termination.
%
% Authors: S. Cipolla, J. Gondzio.
% ==================================================================================================================== %
% Output Structure
OInfo = struct();
% ===============================================================
% Auxilliary Initializations
[m,n] = size(A);
x=xk;
y=yk;
z=zk;
e_pos_vars = ones(num_of_pos_vars,1);       % Vector of ones of dimension |C|.
% ==================================================================================================================== %  
% Initialize parameters
% -------------------------------------------------------------------------------------------------------------------- %
iter = 0;
alpha_x = 0;     % Step-length for primal variables (initialization)
alpha_z = 0;     % Step-length for dual variables (initialization)
sigmamin = 0.05; % Heuristic value.
sigmamax = 0.95; % Heuristic value.
sigma = 0;
OInfo.opt = 0;

if (num_of_pos_vars > 0)                             % Defined only when non-negativity constraints are present.
    mu = (x(pos_vars)'*z(pos_vars))/num_of_pos_vars; % Initial value of mu.
    res_mu = zeros(n,1);
else
    mu = 0;     % Switch to a pure PMM method (no inequality constraints).
    res_mu = [];
end
header(pl);     % Set the printing choice.

if (pc == 1)
    retry = 0;  % Num of times a factorization is re-built (for different regularization values)
else
    retry_p = 0;
    retry_c = 0;
end
max_tries = 50; % Maximum number of times before exiting with an ill-conditioning message.
mu_prev = 0;
reg_limit  = 1e-1*rho;
% ==================================================================================================================== %
while (iter < maxit)
% -------------------------------------------------------------------------------------------------------------------- %
% IP-PMM Main Loop structure:
% Until (PPM related residual < tol) do
%   Choose sigma in [sigma_min, sigma_max] and solve:
%
% [ (Q + Theta^{-1} + rho I)   -A^T ](Delta x)   -(c + Qx_k - A^Ty_k -[z_k] + rho (x-xk))+ X_k^-1(sigma*mu-Z_kX_k)
% [         A               delta I ](Delta y) = -(Ax_k + delta (y-yk) - b)
%
% with suitable modifications when the constarined set is smaller that the
% whole ste of variables.
%   Find two step-lengths a_x, a_z in (0,1] and update:
%       x_{k+1} = x_k + a_x Delta x, y_{k+1} = y_k + a_z Delta y, z_{k+1} = z_k + a_z Delta z
%   k = k + 1
% End
% -------------------------------------------------------------------------------------------------------------------- %
    if (iter > 1)
        res_p  = new_res_p;
        res_d  = new_res_d;
        res_n =  new_res_n;
    else
        F     = Primal_Dual_Res(x,y,z,pos_vars,c,A,b,Q,mu,xk,yk,rho, delta);
        res_d = -F{1};                   % Regularized dual residual.
        res_p = -F{2};                   % Regularized primal residual.
        res_n = natural_prox_res(c,A,b,Q,x,y,pos_vars,xk,yk,rho,delta);
    end
    % ================================================================================================================ %
    % Check termination criteria
    % ---------------------------------------------------------------------------------------------------------------- %
    
     if res_n < 10^4*tol*min(1,norm(x-xk)+norm(y-yk))   
       OInfo.IPMIter= iter; 
       OInfo.opt    = 1;
       break;
    end  
    % ================================================================================================================ %
    iter = iter+1;
    % ================================================================================================================ %
    % ================================================================================================================ %
    % Compute the Newton factorization.
    % ---------------------------------------------------------------------------------------------------------------- %
    pivot_thr = reg_limit;
    NS = Newton_factorization(A,A_tr,Q,x,z,delta,rho,pos_vars,free_variables,pivot_thr);
    % ================================================================================================================ %
   switch pc
        case 1 % No predictor-corrector. 
        % ============================================================================================================ %
        % Compute the parameter sigma and based on the current solution
        % ------------------------------------------------------------------------------------------------------------ %
        if (iter > 1)
            sigma = max(1-alpha_x,1-alpha_z)^5;
        else
            sigma = 0.5;
        end

        sigma = min(sigma,sigmamax);
        sigma = max(sigma,sigmamin);
        % ============================================================================================================ %
        if (num_of_pos_vars > 0)
            res_mu(pos_vars) = (sigma*mu).*e_pos_vars - x(pos_vars).*z(pos_vars);
        end
        % ============================================================================================================ %
        % Solve the Newton system and calculate residuals.
        % ------------------------------------------------------------------------------------------------------------ %
        [dx,dy,dz,instability] = Newton_backsolve(NS,res_p,res_d,res_mu,pos_vars,free_variables);
        if (instability == true) % Checking if the matrix is too ill-conditioned. Mitigate it.
            if (retry < max_tries)
                fprintf('The system is re-solved, due to bad conditioning.\n')
                iter = iter -1;
                retry = retry + 1;
                reg_limit = max(reg_limit*10,tol);
                continue;
            else
                fprintf('The system matrix is too ill-conditioned.\n');
                OInfo.opt = 2;
                OInfo.IPMIter = iter;
                break;
            end
        end
        % ============================================================================================================ %
        case 2 % Mehrotra predictor-corrector. ONLY when num_of_pos_vars > 0!!
    % ================================================================================================================ %
    % Predictor step: Set sigma = 0. Solve the Newton system and compute a centrality measure.
    % ---------------------------------------------------------------------------------------------------------------- %
        res_mu(pos_vars) = - x(pos_vars).*z(pos_vars);
        % ============================================================================================================ %
        % Solve the Newton system with the predictor right hand side -> Optimistic view, solve as if you wanted to 
        %                                                               solve the original problem in 1 iteration.
        % ------------------------------------------------------------------------------------------------------------ %
        [dx,dy,dz,instability] = Newton_backsolve(NS,res_p,res_d,res_mu,pos_vars,free_variables);
        if (instability == true) % Checking if the matrix is too ill-conditioned. Mitigate it.
            if (retry_p < max_tries)
                fprintf('The system is re-solved, due to bad conditioning  of predictor system.\n')
                iter = iter -1;
                retry_p = retry_p + 1;
                reg_limit = min(0.5,max(reg_limit*10,tol));
                continue;
            else
                fprintf('The system matrix is too ill-conditioned.\n');
                OInfo.opt = 2;
                OInfo.IPMIter = iter;
                break;
            end
        end
        retry = 0;
        % ============================================================================================================ %
        
        % ============================================================================================================ %
        % Step in the non-negativity orthant.
        % ------------------------------------------------------------------------------------------------------------ %
        idx = false(n,1);
        idz = false(n,1);
        idx(pos_vars) = dx(pos_vars) < 0; % Select all the negative dx's (dz's respectively)
        idz(pos_vars) = dz(pos_vars) < 0;     
        alphamax_x = min([1;-x(idx)./dx(idx)]);
        alphamax_z = min([1;-z(idz)./dz(idz)]);
        tau = 0.995;
        alpha_x = tau*alphamax_x;
        alpha_z = tau*alphamax_z;
        % ============================================================================================================ %
        centrality_measure = (x(pos_vars) + alpha_x.*dx(pos_vars))'*(z(pos_vars) + alpha_z.*dz(pos_vars));
        mu = (centrality_measure/(num_of_pos_vars*mu))^2*(centrality_measure/num_of_pos_vars);
    % ================================================================================================================ %
        
    % ================================================================================================================ %
    % Corrector step: Solve Newton system with the corrector right hand side. Solve as if you wanted to direct the 
    %                 method in the center of the central path.
    % ---------------------------------------------------------------------------------------------------------------- %
        res_mu(pos_vars) = mu.*e_pos_vars - dx(pos_vars).*dz(pos_vars);
        % ============================================================================================================ %
        % Solve the Newton system with the predictor right hand side -> Optimistic view, solve as if you wanted to 
        %                                                               solve the original problem in 1 iteration.
        % ------------------------------------------------------------------------------------------------------------ %
        [dx_c,dy_c,dz_c,instability] = Newton_backsolve(NS,zeros(m,1),zeros(n,1),res_mu,pos_vars,free_variables);
        if (instability == true) % Checking if the matrix is too ill-conditioned. Mitigate it.
            if (retry_c < max_tries)
                fprintf('The system is re-solved, due to bad conditioning of corrector.\n')
                iter = iter -1;
                retry_c = retry_c + 1;
                mu = mu_prev;
                reg_limit = min(0.5,max(reg_limit*10,tol));
                continue;
            else
                fprintf('The system matrix is too ill-conditioned, increase regularization.\n');
                OInfo.opt = 2;
                OInfo.IPMIter = iter;
                break;
            end
        end
        % ============================================================================================================ %
        dx = dx + dx_c;
        dy = dy + dy_c;
        dz = dz + dz_c; 
    
     % ============================================================================================================ %
       case 3 % Multiple Corrections. ONLY when num_of_pos_vars > 0!!
    % ================================================================================================================ %
        if (iter > 1) || min(alpha_x,alpha_z)>0.5
            mu_target = 0.1*mu;
            factor    = 0.05;
        else
            mu_target = 0.7*mu;
            factor    = 0.1;
        end
    % Predictor step
    % ---------------------------------------------------------------------------------------------------------------- %
        res_mu(pos_vars) = - x(pos_vars).*z(pos_vars) + factor*mu_target;
        % ============================================================================================================ %
        % Solve the Newton system with the predictor right hand side 
        % ------------------------------------------------------------------------------------------------------------ %
        [dx,dy,dz,instability] = Newton_backsolve(NS,res_p,res_d,res_mu,pos_vars,free_variables);
        if (instability == true) % Checking if the matrix is too ill-conditioned. Mitigate it.
            if (retry_p < max_tries)
                fprintf('The system is re-solved, due to bad conditioning  of predictor system.\n')
                iter = iter -1;
                retry_p = retry_p + 1;
                reg_limit = min(0.5,max(reg_limit*10,tol));
                continue;
            else
                fprintf('The system matrix is too ill-conditioned.\n');
                OInfo.opt = 2;
                OInfo.IPMIter = iter;
                break;
            end
        end
        retry = 0;
        % ============================================================================================================ %
        
        % ============================================================================================================ %
        % Step in the non-negativity orthant.
        % ------------------------------------------------------------------------------------------------------------ %
        idx = false(n,1);
        idz = false(n,1);
        idx(pos_vars) = dx(pos_vars) < 0; % Select all the negative dx's (dz's respectively)
        idz(pos_vars) = dz(pos_vars) < 0;     
        alphamax_x = min([1;-x(idx)./dx(idx)]);
        alphamax_z = min([1;-z(idz)./dz(idz)]);
        tau = 0.995;
        alpha_x = tau*alphamax_x;
        alpha_z = tau*alphamax_z;
        % ================================================================================================================ %
        K_c =10;
        bmin=0.001; bmax =1000; delta_a=0.2; 
        for i = 1: K_c
            % Computing Trial Point
            talpha_x = min(1,1.5*alpha_x+delta_a); 
            talpha_z = min(1,1.5*alpha_z+delta_a); 
            xt                = x + talpha_x*dx; 
            zt                = z + talpha_z*dz; 
            res_mu(pos_vars)  = xt(pos_vars).*zt(pos_vars); 
            res_mu(pos_vars)  = min(max(bmin*mu_target,res_mu(pos_vars)), bmax*mu_target) - res_mu(pos_vars);
            res_mu(res_mu < - bmax*mu_target)= -bmax*mu_target;
            res_mu(res_mu > 2*bmax*mu_target)=  2*bmax*mu_target;
            [dx_c,dy_c,dz_c,instability] = Newton_backsolve(NS,zeros(m,1),zeros(n,1),res_mu,pos_vars,free_variables);
            if (instability == true)
                if (retry_p < max_tries)
                    fprintf('The system is re-solved, due to bad conditioning  of corrector system.\n')
                    iter = iter -1;
                    retry_p = retry_p + 1;
                    reg_limit = min(0.5,max(reg_limit*10,tol));
                    continue;
                else
                    fprintf('The system matrix is too ill-conditioned.\n');
                    OInfo.opt = 2;
                    OInfo.IPMIter = iter;
                    break;
                end
            end
            retry_p = 0;
            dxt =  dx  + dx_c;
            dzt =  dz  + dz_c;
            dyt =  dy  + dy_c;
            % Check Improvement
            idxt = false(n,1);
            idzt = false(n,1);
            idxt(pos_vars) = dxt(pos_vars) < 0; % Select all the negative dx's (dz's respectively)
            idzt(pos_vars) = dzt(pos_vars) < 0;     
            alphamax_x = min([1;-x(idxt)./dxt(idxt)]);
            alphamax_z = min([1;-z(idzt)./dzt(idzt)]);
            tau = 0.995;
            alpha_xn = tau*alphamax_x;
            alpha_zn = tau*alphamax_z;
            if (alpha_xn < 1.01*(alpha_x) &&  alpha_zn < 1.01*(alpha_z)) 
                %|| (alpha_xn < 1e-1*(alpha_x))  ||  (alpha_zn < 1e-1*(alpha_z))
                %alpha_x = alpha_xn;
                %alpha_z = alpha_zn;
              break;
            else
            alpha_x = alpha_xn;
            alpha_z = alpha_zn;
            dx =  dxt;
            dz =  dzt;
            dy =  dyt;
            %centrality_measure = (x(pos_vars) + alpha_x.*dx(pos_vars))'*(z(pos_vars) + alpha_z.*dz(pos_vars));
            %mu = (centrality_measure/(num_of_pos_vars*mu))^2*(centrality_measure/num_of_pos_vars);
            end
        end

        % ============================================================================================================ %
%         dx = dx + dx_c;
%         dy = dy + dy_c;
%         dz = dz + dz_c; 
        
   end

    % ================================================================================================================ %
    % Compute the new iterate:
    % Determine primal and dual step length. Calculate "step to the boundary" alphamax_x and alphamax_z. 
    % Then choose 0 < tau < 1 heuristically, and set step length = tau * step to the boundary.
    % ---------------------------------------------------------------------------------------------------------------- %
    if (num_of_pos_vars > 0)
        idx = false(n,1);
        idz = false(n,1);
        idx(pos_vars) = dx(pos_vars) < 0; % Select all the negative dx's (dz's respectively)
        idz(pos_vars) = dz(pos_vars) < 0;       
        alphamax_x = min([1;-x(idx)./dx(idx)]);
        alphamax_z = min([1;-z(idz)./dz(idz)]);        
        tau  = 0.995;
        alpha_x = tau*alphamax_x;
        alpha_z = tau*alphamax_z;
    else
        alpha_x = 1;         % If we have no inequality constraints, Newton method is exact -> Take full step.
        alpha_z = 1;
    end
    % ================================================================================================================ %    
    
    % ================================================================================================================ %
    % Make the step.
    % ---------------------------------------------------------------------------------------------------------------- %
    x = x+alpha_x.*dx; y = y+alpha_z.*dy; z = z+alpha_z.*dz;
    if (num_of_pos_vars > 0) % Only if we have non-negativity constraints.
        mu_prev = mu;
        mu = (x(pos_vars)'*z(pos_vars))/num_of_pos_vars;
        mu_rate = abs((mu-mu_prev)/max(mu,mu_prev));
    end
    % ================================================================================================================ %
    % Computing the new residuals.
    % ================================================================================================================ %
    F     = Primal_Dual_Res(x,y,z,pos_vars,c,A,b,Q,mu,xk,yk,rho, delta);
    new_res_d = -F{1};                   % Regularized dual residual.
    new_res_p = -F{2};                   % Regularized primal residual.
    new_res_n = natural_prox_res(c,A,b,Q,x,y,pos_vars,xk,yk,rho,delta);
    % ================================================================================================================ %
    % Print iteration output.  
    % ---------------------------------------------------------------------------------------------------------------- %
    pres_inf = norm(new_res_p);
    dres_inf = norm(new_res_d);  
    output(pl,iter,pres_inf,dres_inf,mu,sigma,alpha_x,alpha_z);
    % ================================================================================================================ %
end % while (iter < maxit)

if iter == maxit  
   OInfo.IPMIter=maxit;
   OInfo.opt = 0;
end


% The IPM has terminated because the solution accuracy is reached or the maximum number 
% of iterations is exceeded. Print result. 

if (pl >0 )
    fprintf('iterations: %4d\n', iter);
    fprintf('primal feasibility: %8.2e\n', norm(res_p));
    fprintf('dual feasibility: %8.2e\n', norm(res_d));
    fprintf('complementarity: %8.2e\n', full(dot(x,z)/n));  
end
end


% ==================================================================================================================== %
% header + output printing functions: 
% pl = 1: primal-dual infeasibility and mu is printed at each iteration k
% pl = 2: primal-dual infeasibility, mu, sigma, and step-lengths are printed at each iteration k
% -------------------------------------------------------------------------------------------------------------------- %
function header(pl)
    if (pl >= 1)
        fprintf(' ');
        fprintf('%4s    ', 'iter');
        fprintf('%8s  ', 'pr feas');
        fprintf('%8s  ', 'dl feas');
        fprintf('%8s  ', 'mu');
    end
    if (pl >= 2)
        fprintf('  ');
        fprintf('%8s  ', 'sigma');
        fprintf('%8s  ', 'alpha_x');
        fprintf('%8s  ', 'alpha_z');
    end
    if (pl >= 1)
        fprintf('\n ====    ========  ========  ========');
    end
    if (pl >= 2)
        fprintf('    ========  ========  ========');
    end
    if (pl >= 1) fprintf('\n'); end
end


function output(pl,it,xinf,sinf,mu,sigma,alpha_x,alpha_z)
    if (pl >= 1)
        fprintf(' ');
        fprintf('%4d    ', it);
        fprintf('%8.2e  ', xinf);
        fprintf('%8.2e  ', sinf);
        fprintf('%8.2e  ', mu);
    end
    if (pl >= 2)
        fprintf('  ');
        fprintf('%8.2e  ', sigma);
        fprintf('%8.2e  ', alpha_x);
        fprintf('%8.2e  ', alpha_z);
    end
    if (pl >= 1) fprintf('\n'); end
end
% ==================================================================================================================== %
% ******************************************************************************************************************** %
% END OF FILE
% ******************************************************************************************************************** %
