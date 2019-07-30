for i=0:199
    load(sprintf('test_scan_%.3d_dist.mat',i))
    M = dist_map < 500; %The mask indicates if the vertices are connected or not (scans can have few connected components)
    save(sprintf('test_scan_%.3d_mask.mat',i),'M')
end