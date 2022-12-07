function [F] = Primal_Dual_Res(x,y,z,pos_vars,...
                      g,A,b,H,mu,...
                      xk,yk,rho, delta)
F    = cell(2,1);
F{1} = H*x+g-A.'*y-z+rho*(x-xk); % xi_d                           
F{2} = A*x-b+delta.*(y-yk);      % xi_p 
%F{3} = z(pos_vars).*x(posvars)-mu;
end

