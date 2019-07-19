function calc_dist_matrix_shape_collection(path_shapes,path_distance_matrix,num_workers)
    addpath(genpath('./../../../Tools/'))
    d = dir([path_shapes,'*.mat']);
    
    if ~exist(path_distance_matrix, 'dir')
       mkdir(path_distance_matrix)
    end
    
    myCluster = parcluster;
    myCluster.NumWorkers = num_workers;
    parpool(num_workers);

    parfor i=1:numel(d)
        if ~isfile([path_distance_matrix,d(i).name])
            try
                S=load([path_shapes,d(i).name]);
                D = calc_dist_matrix(S); D = single(D);
                parsave([path_distance_matrix,d(i).name],D);
                display(i)
            catch
                display(d(i).name)
            end
        end
    end
    
    delete(gcp('nocreate'))
end

