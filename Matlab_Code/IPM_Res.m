function [nr_res_p,nr_res_d,mu] = IPM_Res(c,A,A_tr,b,Q,x,y,z,pos_vars,num_of_pos_vars)
nr_res_p = b-A*x;                 % Non-regularized primal residual
nr_res_d = c-A_tr*y-z + Q*x;      % Non-regularized dual residual
if num_of_pos_vars > 0
   mu = (x(pos_vars)'*z(pos_vars))/num_of_pos_vars;
else
   mu = 0 ; 
end

