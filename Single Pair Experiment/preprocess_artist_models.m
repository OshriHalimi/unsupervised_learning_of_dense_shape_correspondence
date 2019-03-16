clear all; close all; clc

addpath(genpath('./'))
addpath(genpath('./../Tools/'))

global_params.num_evecs = 150;
shot_params.num_bins = 10;
shot_params.radius = 0.5;

%% Preprocess Model 0
model = load('./artist_models/model_0_remesh'); model = model.part;
model.X = model.VERT(:,1); model.Y = model.VERT(:,2); model.Z = model.VERT(:,3);
figure; trisurf(model.TRIV,model.VERT(:,1),model.VERT(:,2),model.VERT(:,3)); axis equal; 

[model_evecs,~,model_evals,model_S] = extract_eigen_functions_new(model,global_params.num_evecs);
model_shot = calc_shot(model.VERT', model.TRIV', 1:model.n, shot_params.num_bins, shot_params.radius, 3)';
%save as single
model_evecs_trans = single(model_evecs'*model_S);
model_evecs = single(model_evecs);
model_S = single(full(diag(model_S)));
save(['./tf_artist/model_0.mat'],'model_shot', 'model_evecs', 'model_evecs_trans', 'model_S', 'shot_params');
    
D = calc_dist_matrix(model);
D = single(D);
save(['./tf_artist/model_0_dist.mat'],'D');

%% Preprocess Model 1
model = load('./artist_models/model_1_remesh'); model = model.model;
model.X = model.VERT(:,1); model.Y = model.VERT(:,2); model.Z = model.VERT(:,3);
figure; trisurf(model.TRIV,model.VERT(:,1),model.VERT(:,2),model.VERT(:,3)); axis equal; 

[model_evecs,~,model_evals,model_S] = extract_eigen_functions_new(model,global_params.num_evecs);
model_shot = calc_shot(model.VERT', model.TRIV', 1:model.n, shot_params.num_bins, shot_params.radius, 3)';
%save as single
model_evecs_trans = single(model_evecs'*model_S);
model_evecs = single(model_evecs);
model_S = single(full(diag(model_S)));
save(['./tf_artist/model_1.mat'],'model_shot', 'model_evecs', 'model_evecs_trans', 'model_S', 'shot_params');
    
D = calc_dist_matrix(model);
D = single(D);
save(['./tf_artist/model_1_dist.mat'],'D');