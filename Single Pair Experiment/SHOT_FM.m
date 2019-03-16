%% FM Comparison (with SHOT descriptors) - First run preprocess_artist_model.m
clear all; close all; clc

addpath(genpath('./'))
addpath(genpath('./../Tools/'))

model_0 = load('./tf_artist/model_0.mat');
model_1 = load('./tf_artist/model_1.mat');
A = model_0.model_evecs_trans*model_0.model_shot;
B = model_1.model_evecs_trans*model_1.model_shot;
C = mldivide(A',B')'; %B = C*A
P = model_1.model_evecs*C*model_0.model_evecs_trans;
P = normc(P);
[~, matches_0_1] = max(P,[],1);

mesh_0 = load('./artist_models/model_0_remesh'); mesh_0 = mesh_0.part;
mesh_1 = load('./artist_models/model_1_remesh'); mesh_1 = mesh_1.model;
D_0 = load('./tf_artist/model_0_dist.mat'); D_0 = D_0.D;
D_1 = load('./tf_artist/model_1_dist.mat'); D_1 = D_1.D;

colors = create_colormap(mesh_1,mesh_1);
figure;
subplot(1,2,1); colormap(colors);
plot_scalar_map(mesh_1,[1: size(mesh_1.VERT,1)]');freeze_colors;title('Target');

subplot(1,2,2); colormap(colors(matches_0_1,:));
plot_scalar_map(mesh_0,[1: size(mesh_0.VERT,1)]');freeze_colors;title('Source');

save('matches_axiomatic_SHOT_FM.mat','matches_0_1');

%% PMF Refinement
[~, matches_filter] = ...
    my_mfilter(mesh_0.n, mesh_1.n, matches_0_1 , [], 150000/400, D_0, D_1, 0); %corr is not used and set to [];
   
figure;
subplot(1,2,1); colormap(colors);
plot_scalar_map(mesh_1,[1: size(mesh_1.VERT,1)]');freeze_colors;title('Target');

subplot(1,2,2); colormap(colors(matches_filter,:));
plot_scalar_map(mesh_0,[1: size(mesh_0.VERT,1)]');freeze_colors;title('Source');

save('matches_axiomatic_SHOT_FM_PMF.mat','matches_filter');
