function c = bary_to_ind(M, baryc_file)

    fid = fopen(baryc_file);
    baryc = fscanf(fid, '%f %f %f %f\n');
    fclose(fid);    
    baryc = reshape(baryc,4,[])';
    
    
    c = M.TRIV(baryc(:,1),:);
    [~, ind] = min(baryc(:,2:4),[],2);
    ind = sub2ind(size(c),[1:size(c,1)]', ind);
    c = c(ind);
    
%     N = loadoff_colorFix('Y:\litany\FM_net\Data\shrec_dog_holes\train\holes_dog_shape_1.off');
%     h = figure;         
%     N = reformat_model(N);
%     M = reformat_model(M);
%     colors = create_colormap(M,M);
%     subplot(121), colormap(colors), plot_scalar_map(M, 1:numel(M.X)), freeze_colors
%     subplot(122), colormap(colors(c,:)), plot_scalar_map(N, 1:numel(N.X)), title('filtered')

end