%% SGMDS Comparison - First run preprocess_artist_model.m
clear all; close all; clc

addpath(genpath('./'))
addpath(genpath('./../Tools/'))

surface_1 = load('./artist_models/model_0_remesh'); surface_1 = surface_1.part;
surface_2 = load('./artist_models/model_1_remesh'); surface_2 = surface_2.model;

nv = size(surface_1.X, 1);

%% Calculate LBO Eigenfunctionsand & Eigenvalues
G1=metric_scale(surface_1.X, surface_1.Y, surface_1.Z, surface_1.TRIV,0); 
G2=metric_scale(surface_2.X, surface_2.Y, surface_2.Z, surface_2.TRIV,0); 
[W1, A1]=laplace_beltrami_from_grad(surface_1.TRIV,G1); 
[W2, A2]=laplace_beltrami_from_grad(surface_2.TRIV,G2); 

A1=spdiags(A1,0,nv,nv);  %transform A from col to diag matrix, but with sparse representation.
A2=spdiags(A2,0,nv,nv);  

num_vec = 150;
[Phi1,lambda1]=eigs(W1,A1,num_vec,'SM',struct('disp',0)); %solves W*Phi = A*Phi*Lambda
[Phi2,lambda2]=eigs(W2,A2,num_vec,'SM',struct('disp',0)); %solves W*Phi = A*Phi*Lambda

lambda1_d = diag(lambda1);
lambda2_d = diag(lambda2);
[lambda1_d_s, Idx1] =  sort(lambda1_d);
[lambda2_d_s, Idx2] =  sort(lambda2_d);
Phi1 = Phi1(:, Idx1);
Phi2 = Phi2(:, Idx2);
lambda1 = diag(lambda1_d_s);
lambda2 = diag(lambda2_d_s);

%% Constraints (SHOT Descriptors)
X1 = load('model_0.mat'); H1 = X1.model_shot; 
X2 = load('model_1.mat'); H2 = X2.model_shot; 

%% Spectral Decomposition of Distance Matrix
miu = 10;
D1 = load('./tf_artist/model_0_dist.mat'); D1 = double(D1.D);
D2 = load('./tf_artist/model_1_dist.mat'); D2 = double(D2.D);

tic % Start time measurement of SGMDS algorithm
Alpha1 = get_alpha(D1, 1:nv, Phi1, lambda1, miu);
Alpha2 = get_alpha(D2, 1:nv, Phi2, lambda2, miu);

fact_Alpha=max(norm(Alpha1), norm(Alpha2));
Alpha1 = Alpha1/fact_Alpha;
Alpha2 = Alpha2/fact_Alpha;

%% Solve SGMDS Problem
opts.alpha1=Alpha1;
opts.alpha2=Alpha2;
opts.F1=Phi1'*A1*H1;
opts.F2=Phi2'*A2*H2;
opts.mu1 = 1;
opts.mu2 = 1;
x = find_opt_alpha(opts);
x2=reshape(x,num_vec-1,num_vec-1);
C_phi=zeros(num_vec,num_vec);
C_phi(1)=1;
C_phi(2:end,2:end)=x2;

Y1 = Phi1;
Y2 = Phi2*C_phi;
corr_sgmds = knnsearch(Y2, Y1);
toc % Finish time measurement of SGMDS algorithm
%% Visualization
colors = create_colormap(surface_2,surface_2);
figure;
subplot(1,2,1); colormap(colors);
plot_scalar_map(surface_2,[1: size(surface_2.VERT,1)]');freeze_colors;title('Target');
subplot(1,2,2); colormap(colors(corr_sgmds,:));
plot_scalar_map(surface_1,[1: nv]');freeze_colors;title('Source');

save('matches_axiomatic_SGMDS.mat','corr_sgmds')

%% PMF Refinement
tic % Start time measurement of PMF algorithm
[~, matches_filter] = ...
    my_mfilter(surface_1.n, surface_2.n, corr_sgmds , [], 150000/400, D1, D2, 0);
toc % Finish time measurement of PMF algorithm
%% Visualization
colors = create_colormap(surface_2,surface_2);
figure;
subplot(1,2,1); colormap(colors);
plot_scalar_map(surface_2,[1: size(surface_2.VERT,1)]');freeze_colors;title('Target');
subplot(1,2,2); colormap(colors(matches_filter,:));
plot_scalar_map(surface_1,[1: nv]');freeze_colors;title('Source');

save('matches_axiomatic_SGMDS_PMF.mat','matches_filter')
