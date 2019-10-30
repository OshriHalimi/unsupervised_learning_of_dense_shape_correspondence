function matches_refined = refine_matches(...
    part, model, part_ind, model_ind, C_init, options)

F = part_ind;
G = model_ind;

k = options.k;

W = zeros(k);
for i=1:k
    for j=1:k
        slope = 1;
        direction = [1 slope];
        direction = direction./norm(direction);
        W(i,j) = exp(-0.03*sqrt(i.^2 + j.^2))*norm(cross([direction 0], [i,j, 0]-[1 1 0]));
    end
end
d=ones(1,k);
D = repmat(d,k,1);

mu1 = 1e-2; % diagonal mask
mu2 = 1e1; % orthogonality

for iter=1:options.refine_iters
    
    A = part.evecs'*part.S*F;
    B = model.evecs'*model.S*G;
    
    manifold = euclideanfactory(k,k);
    problem = {};
    
    problem.M = manifold;
    
    problem.cost = @(C) (...
        sum(sum((C*A-B).^2).^0.5) + ...
        mu1 * norm(C.*W,'fro')^2 + ...
        mu2 * (norm(C'*C,'fro')^2 - sum(diag(C'*C).^2) + sum((diag(C'*C) - d').^2) ));
    
    problem.egrad = @(C) (...
        norm_21_gradient(C,A,B) + ...
        mu1 * 2 * C.*W.*W + ...
        mu2 * 4*(C*C'*C - C.*D ));
    
    options.verbosity = 2;
%     options.maxiter = 5e3;
%     C_refined = conjugategradient(problem, C_init, options);
    options.maxiter = 3e2;
    C_refined = trustregions(problem, C_init, options);
    
%     figure,colormap(bluewhitered)
%     subplot(121),imagesc(C_init),colorbar,axis image
%     subplot(122),imagesc(C_refined),colorbar,axis image
    
    [matches_refined, ~] = flann_search(...
        model.evecs', ...
        C_refined*part.evecs', ...
        1, struct());
    
%     [matches_init, ~] = flann_search(...
%         model.evecs', ...
%         C_init*part.evecs', ...
%         1, struct());
    
%     colors = create_colormap(model,model);
%     figure
%     subplot(231), colormap(colors), plot_scalar_map(model, 1:model.n), axis off, view([0 90]), freeze_colors
%     subplot(232), colormap(colors(matches_init,:)), plot_scalar_map(part, 1:part.n), axis off, view([0 90]), freeze_colors
%     subplot(233), colormap(colors(matches_refined,:)), plot_scalar_map(part, 1:part.n), axis off, view([0 90])
%     subplot(234), colormap(colors), plot_scalar_map(model, 1:model.n), axis off, view([-180 -90]), freeze_colors
%     subplot(235), colormap(colors(matches_init,:)), plot_scalar_map(part, 1:part.n), axis off, view([-180 -90]), freeze_colors
%     subplot(236), colormap(colors(matches_refined,:)), plot_scalar_map(part, 1:part.n), axis off, view([-180 -90])
    
    C_init = C_refined;
    
    fps = fps_euclidean(part.VERT, 1e3, randi(part.n));
    F = sparse(fps, 1:length(fps), 1, part.n, length(fps));
    G = sparse(matches_refined(fps), 1:length(fps), 1, model.n, length(fps));
    
end

end
