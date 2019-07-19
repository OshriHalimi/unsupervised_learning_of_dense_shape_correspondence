clear all; close all; clc
addpath(genpath('./faust_synthetic/'))
path_shapes = './faust_synthetic/shapes/';
path_distance_matrix = './faust_synthetic/distance_matrix/';
num_workers = 10; 

calc_dist_matrix_shape_collection(path_shapes,path_distance_matrix,num_workers);