clear all; close all; clc
addpath(genpath('./faust_synthetic/'))
addpath(genpath('./../Tools/'))

path_shapes = './faust_synthetic/shapes/';
path_save = './faust_synthetic/network_data/';

global_params.num_evecs = 120;
shot_params.num_bins = 10; shot_params.radius = 0.1;
preprocess_shape_collection(path_shapes,path_save, global_params, shot_params);