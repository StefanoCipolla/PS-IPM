%This script loads various NETLIB problems and solves them using
%Dual_Regularized IPM
clear all;
clc;
%The path on which all the netlib problems lie
QP_problems_path = './QP_PROBLEMS/QPset/maros'; 

%Finds all the Netlib problems and stores their names in a struct
d = dir(fullfile(QP_problems_path,'*.mat')); 

%Each indice i=1..num_of_netlib_files gives the name of each netlib problem
%though d(i).name

%Open the file to write the results
fileID = fopen('QP_problems_tabular_format_final_results_PPM_MC.txt','a+');
fprintf(fileID,'Problem & PPM Iter & IPM iter (tot) &  Time & Obj Val & Exit Fl. \n'); 
fileID1 = fopen('QP_problems_performance_prof_time_PPM_MC.txt','a+');
fileID2 = fopen('QP_problems_performance_prof_iter_PPM_MC.txt','a+');
fileID3 = fopen('QP_problems_performance_prof_objval_PPM_MC.txt','a+');


model = struct();
fields = {'H','name','xl','xu','al','au','g','g0','A'};
total_iters = 0;
total_time = 0;
total_IPM_iters =0;
scaling_option = 3;
scaling_direction = 'r';
tol = 1e-6;
pc_mode = 3;
print_mode = 3;
rf = 1e-3;
problems_converged = 0;
for k = 1:length(d)% [45,53]
    if (isfield(model,fields)) %If any of the fields is missing, dont remove anything
        model = rmfield(model,fields); %Remove all fields before loading new ones
    end
    model = load(fullfile(QP_problems_path,d(k).name));

    n = size(model.A,2);
    m = size(model.A,1);
  
    [model,b,free_variables,objective_const_term] = QP_Convert_to_Standard_Form(model);
    
    model.name
    
    n_new = size(model.A,2);
    m_new = size(model.A,1);
    model.H = [model.H sparse(n,n_new -n)]; 
    model.H = [model.H ;sparse(n_new-n,n_new)];
    D = Scale_the_problem(model.A,scaling_option,scaling_direction);
    if (scaling_direction == 'l')
        model.A = spdiags(D,0,m_new,m_new)*model.A;  % Apply the left scaling.
        b = b.*D;
    elseif (scaling_direction == 'r')
        model.A = model.A*spdiags(D,0,n_new,n_new);
        model.g = model.g.*D;
        model.H = spdiags(D,0,n_new,n_new)*model.H*spdiags(D,0,n_new,n_new);
    end
    n = n_new;
    m = m_new;
    
    IterStruct=struct();
    rho   = rf*max(tol*(1/max(norm(model.A,'inf'),norm(model.H,'inf'))),10^(-8));
    delta = rho;
    time  = 0; 

    scale = max([1,norm(model.A,'inf'), norm(b,1),norm(model.g,1),norm(model.H,'inf')]);
    


    tic;
    [x,y,z,Info] = PPM_IPM(model.g,model.A,b,model.H,free_variables,tol,200,...
                               pc_mode,print_mode,IterStruct,rho,delta,scale); 
    time = time + toc;
    total_time = total_time + time;
    opt     = Info.opt
    iter    = Info.ExIt;
    IPMiter = Info.IPM_It;
    total_IPM_iters = total_IPM_iters+IPMiter;
    total_iters     = total_iters + iter; % PPM Iters
    obj_val = model.g'*x + objective_const_term + model.g0 + (1/2)*(x'*(model.H*x));
    if (opt == 1)
       problems_converged = problems_converged + 1;
       fprintf(fileID,'%s & %d & %d & %f & %f & %d & opt \n',model.name, iter, IPMiter, time, obj_val,rho); 
     %   fprintf(fileID,'The optimal solution objective is %d.\n',obj_val);
       fprintf(fileID1,'%f \n', time);
       fprintf(fileID2,'%f \n', IPMiter);
       fprintf(fileID3,'%f \n', obj_val);
    else
      fprintf(fileID,'%s & %d & %d & %f & %f & %d &non-opt \n',model.name, iter, IPMiter, time, obj_val,rho);  
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