function plot_mesh(N, color)
    if isfield(N, 'VERT')
        V = N.VERT;
    else
        V = [N.X N.Y N.Z];
    end
    if nargin==1
        trisurf(N.TRIV,V(:,1),V(:,2),V(:,3),zeros(size(V,1),1))
    elseif nargin==2
        trisurf(N.TRIV,V(:,1),V(:,2),V(:,3),'FaceColor',color,'EdgeColor','none')
    end
    axis equal
	xlabel('X')
	ylabel('Y')
	zlabel('Z')
    rotate3d on
    set(gca,'clipping','off')
end
