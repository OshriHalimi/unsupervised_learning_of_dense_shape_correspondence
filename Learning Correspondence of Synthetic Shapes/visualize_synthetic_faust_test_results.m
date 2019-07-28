%% Visualize Unsupervised Network Results
clear all; close all; clc

addpath(genpath('./'))
addpath(genpath('./../Tools/'))

mesh_0 = load('./faust_synthetic/shapes/tr_reg_080'); %Choose the indices of the test pair
mesh_1 = load('./faust_synthetic/shapes/tr_reg_090'); %Choose the indices of the test pair

X = load('./Results/test_faust_synthetic/080_090.mat'); %Choose the indices of the test pair
[~, unsupervised_matches] = max(squeeze(X.softCorr),[],1);

colors = create_colormap(mesh_1,mesh_1);
figure;
subplot(1,2,1); colormap(colors);
plot_scalar_map(mesh_1,[1: size(mesh_1.VERT,1)]');freeze_colors;title('Target');

subplot(1,2,2); colormap(colors(unsupervised_matches,:));
plot_scalar_map(mesh_0,[1: size(mesh_0.VERT,1)]');freeze_colors;title('Source');
