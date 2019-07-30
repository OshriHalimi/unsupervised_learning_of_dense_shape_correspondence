clear all; close all; clc

addpath(genpath('./'))
addpath(genpath('./../Tools/'))

%% Calculate inter pairs
test_idx=[80:99];
pairs_list = [];
for i=1:20
    for j = 1:20
        pairs_list = [pairs_list; test_idx(i),test_idx(j)];
    end
end

delete_idx = floor(pairs_list(:,1)/10) == floor(pairs_list(:,2)/10);
pairs_list(delete_idx,:)=[];

%% Calculate geodesic curve
N_pairs = size(pairs_list,1);
CURVES = zeros(N_pairs,1001);
for i=1:N_pairs
    %here you calculate matches, gt_matches and D_model, for each pair
    source_id = sprintf('%03d', pairs_list(i,1));
    target_id = sprintf('%03d', pairs_list(i,2));
    
    mesh_0 = load(['./faust_synthetic/shapes/tr_reg_',source_id]); %Choose the indices of the test pair
    mesh_1 = load(['./faust_synthetic/shapes/tr_reg_',target_id]); %Choose the indices of the test pair

    X = load(['./Results/test_faust_synthetic/',source_id,'_',target_id,'.mat']); %Choose the indices of the test pair
    [~, matches] = max(squeeze(X.softCorr),[],1);

    D_model = load(['.\faust_synthetic\distance_matrix\tr_reg_',target_id]); %Choose the indices of the test pair
    D_model = D_model.D;

    gt_matches = 1:6890;

    errs = calc_geo_err(matches, gt_matches, D_model);
    curve = calc_err_curve(errs, 0:0.001:1.0)/100;

    %here you calculate the error curves and average them
    errs = calc_geo_err(matches, gt_matches, D_model);
    curve = calc_err_curve(errs, 0:0.001:1.0)/100;
    CURVES(i,:) = curve;
end

avg_curve_unsupervised = sum(CURVES,1)/ N_pairs;
plot(0:0.001:1.0, avg_curve_unsupervised,'r'); set(gca, 'xlim', [0 0.1]); set(gca, 'ylim', [0 1])

plot(0:0.001:1.0, curve); set(gca, 'xlim', [0 0.1]); set(gca, 'ylim', [0 1])

xlabel('Geodesic error')
ylabel('Correspondence Accuracy %')

title('Geodesic error - all inter pairs')