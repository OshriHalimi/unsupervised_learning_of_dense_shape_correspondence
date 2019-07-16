%% Visualize Network Results
clear all; close all; clc

addpath(genpath('./'))
addpath(genpath('./../Tools/'))

mesh_0 = load('./artist_models/model_0_remesh'); mesh_0 = mesh_0.part;
mesh_1 = load('./artist_models/model_1_remesh'); mesh_1 = mesh_1.model;

X = load('./Results/unsupervised_artist_results/model_0_model_1.mat'); 
[~, unsupervised_matches] = max(squeeze(X.softCorr),[],1);

colors = create_colormap(mesh_1,mesh_1);
figure;
subplot(1,2,1); colormap(colors);
plot_scalar_map(mesh_1,[1: size(mesh_1.VERT,1)]');freeze_colors;title('Target');

subplot(1,2,2); colormap(colors(unsupervised_matches,:));
plot_scalar_map(mesh_0,[1: size(mesh_0.VERT,1)]');freeze_colors;title('Source');

%save('matches_unsupervised.mat','unsupervised_matches');