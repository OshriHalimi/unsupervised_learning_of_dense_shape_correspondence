1.Make sure you read the warning at the beginning of demo.m

2. Note there are 3 options:

options.icp_iters = 0;     % 0 for nearest neighbors
options.use_svd   = true;  % false for basic least squares
options.refine_iters = 0;  % 0 for no refinement

For the final challenge we used svd=true and 5 - 10 iterations for icp_iters & refine_iters. 
Generally, the value for icp_iters & refine_iters depends on the quality of the sparse matches.

