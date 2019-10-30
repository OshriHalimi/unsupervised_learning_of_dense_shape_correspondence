function S_tri = calc_tri_areas(M)

xx = cross(M.VERT(M.TRIV(:,3),:)-M.VERT(M.TRIV(:,1),:),M.VERT(M.TRIV(:,2),:)-M.VERT(M.TRIV(:,1),:));
S_tri = 0.5*sqrt(sum(xx.^2,2));

end
