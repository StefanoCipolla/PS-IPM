%This script loads various NETLIB problems and solves them using IP_PMM
clear all;
clc;
%The path on which all the netlib problems lie
Netlib_path = '../NETLIB_PROBLEMS_IN_MATLAB_FORM/netlib'; 
%Finds all the Netlib problems and stores their names in a struct
d = dir(fullfile(Netlib_path,'*.mat')); 


%Open the file to write the results
fileID = fopen('Netlib_tabular_format_final_results_PPM_MC.txt','a+');
fprintf(fileID,'Problem & PPM Iter & IPM iter (tot) &  Time & Obj Val & Exit Fl. \n'); 
fileID1 = fopen('LP_problems_performance_prof_time_PPM_MC.txt','a+');
fileID2 = fopen('LP_problems_performance_prof_iter_PPM_MC.txt','a+');
fileID3 = fopen('LP_problems_performance_prof_objval_PPM.txt','a+');

fields = {'A','obj','sense','rhs','lb','ub','vtype','modelname','varnames','constrnames'};
total_iters = 0;
total_time = 0;
total_IPM_iters =0;
scaling_direction = 'r';
scaling_mode = 3;
pc_mode = 3;
tol = 1e-6;
problems_converged = 0;
print_mode = 3;
rf = 1;
%Each indice k=1..num_of_netlib_files gives the name of each netlib problem through d(i).name
for k = 1:length(d)
    load(fullfile(Netlib_path,d(k).name))
   
    c = model.obj; 
    A = model.A;
    b = model.rhs;
    
    model.modelname
    
    [c,A,b,free_variables,objective_const_term] = LP_Convert_to_Standard_Form(c, A, b, model.lb, model.ub, model.sense);

    n = size(A,2);
    Q = sparse(n,n);
     if (scaling_direction == 'r')
        [D,~] = Scale_the_problem(A,scaling_mode,scaling_direction);
        A = A*spdiags(D,0,n,n); % Apply the right scaling.
        c = c.*D;
    elseif (scaling_direction == 'l')
        [D,~] = Scale_the_problem(A,scaling_mode,scaling_direction);
        A = spdiags(D,0,m,m)*A;  % Apply the left scaling.
        b = b.*D;
    elseif (scaling_direction == 'b')
        [D_R,D_L] = Scale_the_problem(A,scaling_mode,scaling_direction);
        if (size(D_L,1) ~= 0)
            A = (spdiags(D_L.^(1/2),0,m,m)*A)*spdiags(D_R.^(1/2),0,n,n);
            b = b.*D_L;
        else
            A = A*spdiags(D_R,0,n,n); % Apply the right scaling.        
        end
        c = c.*D_R;
     end
    
    IterStruct=struct();
    rho   = rf*max(tol*(1/(norm(A,'inf'))),10^(-8));
    delta = rho;
    time = 0; 

    scale = max([1,norm(A,'inf'), norm(b,1),norm(c,1)]);
    tic;
    [x,y,z,Info] = PPM_IPM(c,A,b,Q,free_variables,tol,200,...
                               pc_mode,print_mode,IterStruct,rho,delta, scale); 
    time = time + toc;
    total_time = total_time + time;
    opt     = Info.opt
    iter    = Info.ExIt;
    IPMiter = Info.IPM_It;
    total_iters     = total_iters + iter; % PPM Iters
    total_IPM_iters = total_IPM_iters+IPMiter;
    obj_val = c'*x + objective_const_term;
    if (opt == 1)
       problems_converged = problems_converged + 1;
      fprintf(fileID,'%s & %d & %d & %f & %f & %d & opt \n',model.modelname, iter, IPMiter, time, obj_val,rho); 
     %   fprintf(fileID,'The optimal solution objective is %d.\n',obj_val);
       fprintf(fileID1,'%f \n', time);
       fprintf(fileID2,'%f \n', IPMiter);
       fprintf(fileID3,'%f \n', obj_val);
    else
       fprintf(fileID,'%s & %d & %d & %f & %f & %d &non-opt \n',model.modelname, iter, IPMiter, time, obj_val,rho);  
       fprintf(fileID1,'inf \n');
       fprintf(fileID2,'inf \n');
       fprintf(fileID3,'inf \n');
    end
    
end
fprintf(fileID,'The total iterates were: %d and the total time was %d. %d problems converged.\n',total_iters,total_time,problems_converged);
fprintf(fileID,'The total PPM iterates were: %d, \nThe total IPPM iterates were: %d, \n The total time was: %d and %d problems converged.\n',...
        total_iters,total_IPM_iters,total_time,problems_converged);
fprintf(fileID,'The reduction factror is rf = %d \n ',rf);
fprintf(fileID,'The Stopping Tol was: %f \n ',tol);
fclose(fileID);
fclose(fileID1);
fclose(fileID2);
fclose(fileID3);



