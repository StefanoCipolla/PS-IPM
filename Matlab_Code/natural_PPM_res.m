function [npres] = natural_PPM_res(g,A,b,H,x,y,pos_var)
r     = cell(2,1);
r{1}  = H*x+g-A.'*y; 
r{1}(pos_var) = x(pos_var)-max(0,x(pos_var)-r{1}(pos_var)); 
r{2}  = A*x-b;  
npres = norm(r{1})+norm(r{2});
end

