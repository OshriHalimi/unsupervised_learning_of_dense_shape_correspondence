function [matches_in, matches_out] = my_mfilter(num_vert_part,num_vert_model, matches_in, corr, var, dist_part, dist_model, verbose)


% var = 100;
% matches_in = knnsearch(model.S*model.evecs, part.S*part.evecs*C_in');
dist_model = sparse(double(dist_model));
dist_part = sparse(double(dist_part));

P_in = sparse(1:num_vert_part, matches_in, 1, num_vert_part, num_vert_model); %corr';
KM = exp(-(dist_model*P_in').^2 ./ (2*var));
KN = exp(-(dist_part).^2 ./ (2*var));

F = KM * KN';
%clear heavy variables
dist_model=[];
dist_part=[];
P_in = [];
KM=[];
KN=[];

matches_out = assignmentProblemAuctionAlgorithm(F*1e5);
P_out = sparse(1:num_vert_model, matches_out, 1, num_vert_model, num_vert_part);

fprintf('our score: %.2f\n', trace(P_out'*F))
% fprintf('gt score: %.2f\n', trace(P_gt'*F))

[matches_out,~] = find(P_out);

%figure, plot(matches_out);

if verbose == 1
    colors = create_colormap(model,model);
    figure
    subplot(121), colormap(colors), plot_scalar_map(model, 1:num_vert_model), freeze_colors
    subplot(122), colormap(colors(matches_out,:)), plot_scalar_map(part, 1:num_vert_part), title('filtered')
end
    



end