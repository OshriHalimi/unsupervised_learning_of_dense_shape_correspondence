function plot_scalar_map(M, f, diffuse_strength)

if nargin==2
    diffuse_strength=0.6;
end

trisurf(M.TRIV,M.VERT(:,1),M.VERT(:,2),M.VERT(:,3),double(f),...
    'SpecularStrength',0.15,...
    'DiffuseStrength',diffuse_strength);
axis equal
shading interp
rotate3d on
set(gca,'clipping','off')
xlabel('X')
ylabel('Y')
zlabel('Z')

end
