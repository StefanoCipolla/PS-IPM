function [npres] = natural_prox_res(g,A,b,H,x,y,pos_var,xk,yk,rho,delta)
r     = cell(2,1);
r{1}  = H*x+g-A.'*y+rho.*(x-xk); 
r{1}(pos_var) = x(pos_var)-max(0,x(pos_var)-r{1}(pos_var)); 
r{2}  = A*x-b+delta.*(y-yk);  
npres = norm(r{1})+norm(r{2});
end