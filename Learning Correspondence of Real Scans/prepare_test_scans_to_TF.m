git% This script runs the experiment for perfect isometry using *SHOT*
function prepare_test_scans_to_TF
clear all;close all;clc;
rng(42);
log_file = {};
to_tensorflow = {};
%% Dependencies
addpath(genpath('./'))
addpath(genpath('./../Tools'))

%% folders
test_scans_folder = './MPI-FAUST/test/scans/'; %PATH to MPI-FAUST data. can be downloaded from http://faust.is.tue.mpg.de/
tf_scans_folder = './tf_faust_scans_eps_2p5_7K/test/';

%% params
global_params.num_evecs = 100;
model_scale = 100;
shot_params.num_bins = 10;
shot_params.radius = 9;
num_points = 7000;

%% prepare test scans 
%
d = dir([test_scans_folder, '*.ply']);
for i = 1:numel(d)        
        filename = split(d(i).name,'.'); filename = filename{1};
        [TRIV,VERT] = plyread([test_scans_folder, d(i).name],'tri');
        model.VERT = VERT;
        model.TRIV = TRIV;
        model.X = VERT(:,1); model.Y = VERT(:,2); model.Z = VERT(:,3);
        model.X = model_scale*model.X; model.Y = model_scale*model.Y; model.Z = model_scale*model.Z;
        [model_fixed, is_outlier] = cleanup(model);

        model_remesh = remesh(model_fixed, struct('vertices',num_points, 'verbose', 1));
        model_remesh.X = model_remesh.VERT(:,1); 
        model_remesh.Y = model_remesh.VERT(:,2);
        model_remesh.Z = model_remesh.VERT(:,3);
        
        if numel(model_remesh.X) == num_points
%             continue
        else
            disp(['found model ' filename])
            delta = 1;
            while numel(model_remesh.X) < num_points
                model_remesh = remesh(model_fixed, struct('vertices',num_points + delta, 'verbose', 0));
                model_remesh.X = model_remesh.VERT(:,1); 
                model_remesh.Y = model_remesh.VERT(:,2);
                model_remesh.Z = model_remesh.VERT(:,3);
                delta = delta + 1
            end
        end
            
                
        [model_evecs,~,model_evals,model_S] = extract_eigen_functions_new(model_remesh,global_params.num_evecs);
        model_shot = calc_shot([model_remesh.X model_remesh.Y model_remesh.Z]', model_remesh.TRIV', 1:numel(model_remesh.X), shot_params.num_bins, shot_params.radius, 3)';
                        
        %save as single
        model_evecs_trans = single(model_evecs'*model_S);
        model_evecs = single(model_evecs);
        model_S = single(full(diag(model_S)));
        
        
        
        
        model_remesh.VERT = [model_remesh.X model_remesh.Y model_remesh.Z]; model_remesh.n = numel(model_remesh.X);
        dist_map = calc_dist_matrix(model_remesh);
        
        mysave1([tf_scans_folder filename],...
            model_shot, model_evecs, model_evecs_trans, model_S, shot_params, dist_map, model_remesh);
                
end

end

function mysave1(location,model_shot, model_evecs, model_evecs_trans, model_S, shot_params, dist_map, model_remesh)
    
    save([location '.mat'],'model_shot', 'model_evecs', 'model_evecs_trans', 'model_S', 'shot_params');
    save([location '_dist' '.mat'],'dist_map');
    save([location '_remesh' '.mat'],'model_remesh');
            
end

