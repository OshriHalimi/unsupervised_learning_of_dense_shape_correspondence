function plot_cloud_color(V, C, sz)

scatter3(V(:,1), V(:,2), V(:,3), sz, C, 'fill')

axis equal
rotate3d on

end
