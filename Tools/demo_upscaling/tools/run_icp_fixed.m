function [D, matches] = run_icp_fixed(M, N, C_init, max_iters)

verbose = 1;

if verbose
    fprintf('Running ICP...\n');
end

ann_params = struct;
% ann_params.algorithm = 'linear'; %use 'kmeans' for some speedup
% ann_params.trees = 8;
% ann_params.checks = 64;
% ann_params.centers_init = 'kmeanspp';
% ann_params.iterations = -1;

kN = size(N.evecs,1);

% flann_search(target, query)
[matches, dists] = flann_search(C_init*M.evecs', N.evecs', 1, ann_params);
% [matches, dists] = knnsearch(N.S*N.evecs, M.S*M.evecs*C_init');

err = sum(sqrt(dists));
err = err / (kN*size(C_init,1));

if verbose
    fprintf('(0) MSE: %.2e\n', err);
end

if max_iters == 0
    D = C_init;
%     matches = matches';
    return
end

% Start iterations

D_prev = C_init;
err_prev = err;
matches_prev = matches;

% vidx = 2:4;
% figure, plot_cloud([],N.evecs(:,vidx),'b.'); axis equal; hold on; plot_cloud([],M.evecs(:,vidx),'r.');
% figure, plot_cloud([],N.evecs(:,vidx),'b.'); axis equal; hold on; pp = M.evecs*C_init';plot_cloud([],pp(:,vidx),'r.');

% [u,~,v] = svd(C_init);
% D = u*v';

for i=1:max_iters
    
%     if i>1
        [U,~,V] = svd((M.evecs(matches,:)'*M.S(matches,matches)) * (N.S*N.evecs));
        D = U * V(:,1:size(C_init,2))';
        D = D';
%     end
    
%     figure, plot_cloud([],N.evecs(:,vidx),'b.'); axis equal; hold on; pp = M.evecs*D';plot_cloud([],pp(:,vidx),'r.');
    
    %     matches = flann_search(M.evecs', D*N.evecs', 1, ann_params);
    [matches, dists] = flann_search(D*M.evecs', N.evecs', 1, ann_params);
%     [matches, dists] = knnsearch(N.S*N.evecs, M.S*M.evecs*D');
    err = sum(sqrt(dists));
    err = err / (kN*size(C_init,1));
    
    if verbose
        fprintf('(%d) MSE: %.2e\n', i, err);
    end
    
    if err > err_prev
        if verbose
            fprintf('Local optimum reached.\n');
        end
        D = D_prev;
        matches = matches_prev;
        break;
    end
    
    if (err_prev - err) < 5e-6
        if verbose
            fprintf('Local optimum reached.\n');
        end
        break;
    end
    
    err_prev = err;
    D_prev = D;
    matches_prev = matches;
    
end

% matches = matches';

end
