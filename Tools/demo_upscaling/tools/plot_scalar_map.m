function h = plot_scalar_map(M, f, diffuse_strength, show_edges)

if nargin<3
    diffuse_strength=0.6;
end

if nargin<4
    show_edges=false;
end

if isfield(M, 'VERT')
    V = M.VERT;
else
    V = [M.X M.Y M.Z];
end

if ~show_edges
    h = trisurf(M.TRIV,V(:,1),V(:,2),V(:,3),double(f),...
        'SpecularStrength',0.15,...
        'DiffuseStrength',diffuse_strength);
else
    h = trisurf(M.TRIV,V(:,1),V(:,2),V(:,3),double(f),...
        'SpecularStrength',0.15,...
        'DiffuseStrength',diffuse_strength,...
        'EdgeColor', 'k');
end
axis equal
if ~show_edges
    shading interp
end
rotate3d on
set(gca,'clipping','off')
xlabel('X')
ylabel('Y')
zlabel('Z')

end
