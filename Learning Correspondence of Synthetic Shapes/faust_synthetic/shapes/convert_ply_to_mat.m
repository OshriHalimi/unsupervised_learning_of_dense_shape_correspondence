%% Run this script in the off directory to convert all the .ply files to .mat files with the same name
clear all; close all
addpath(genpath('./../../../Tools/'))
d = dir('./*.ply')
N_models = length(d);

for i=1:N_models
    model_filename = d(i).name; tokens = split(model_filename,'.'); model_name = tokens{1};
    shape = plyread(model_filename);
    VERT = [shape.vertex.x,shape.vertex.y,shape.vertex.z];
    TRIV = cell2mat(shape.face.vertex_indices) + 1; %+1 due to zero based indexing in the .ply file
    n = size(VERT,1);
    m = size(TRIV,1);
    savename = [model_name,'.mat'];
    save(savename,'VERT','TRIV','n','m')
end
