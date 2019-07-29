%% Calculate geodesic error curves
clear all; close all; clc

addpath(genpath('./'))
addpath(genpath('./../Tools/'))

mesh_0 = load('./faust_synthetic/shapes/tr_reg_080'); %Choose the indices of the test pair
mesh_1 = load('./faust_synthetic/shapes/tr_reg_087'); %Choose the indices of the test pair

X = load('./Results/test_faust_synthetic/080_087.mat'); %Choose the indices of the test pair
[~, matches] = max(squeeze(X.softCorr),[],1);

D_model = load('.\faust_synthetic\distance_matrix\tr_reg_087.mat'); %Choose the indices of the test pair
D_model = D_model.D;

gt_matches = 1:6890;
errs = calc_geo_err(matches, gt_matches, D_model);
curve = calc_err_curve(errs, 0:0.001:1.0)/100;
plot(0:0.001:1.0, curve); set(gca, 'xlim', [0 0.1]); set(gca, 'ylim', [0 1])

xlabel('Geodeisc error')
ylabel('Correspondence Accuracy %')