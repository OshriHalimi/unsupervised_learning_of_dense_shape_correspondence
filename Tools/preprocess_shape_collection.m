function preprocess_shape_collection(path_shapes,path_save,global_params, shot_params)
    d = dir([path_shapes,'*.mat']);
    
    if ~exist(path_save, 'dir')
       mkdir(path_save)
    end

    for i=1:numel(d)
        %Load shapes
        model = load([path_shapes,d(i).name]); 
        model.X = model.VERT(:,1); model.Y = model.VERT(:,2); model.Z = model.VERT(:,3);
        
        %Calculate LBO eigenfunctions and SHOT descriptors
        [model_evecs,~,model_evals,model_S] = extract_eigen_functions_new(model,global_params.num_evecs);
        model_shot = calc_shot(model.VERT', model.TRIV', 1:model.n, shot_params.num_bins, shot_params.radius, 3)';

        %save as single
        model_evecs_trans = single(model_evecs'*model_S);
        model_evecs = single(model_evecs);
        model_S = single(full(diag(model_S)));
        save([path_save,d(i).name],'model_shot', 'model_evecs', 'model_evecs_trans', 'model_S', 'shot_params');

        display(i)
    end
    
    delete(gcp('nocreate'))
end

