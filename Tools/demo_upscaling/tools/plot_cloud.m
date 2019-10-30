function plot_cloud(M, points, style)
    if isempty(points)
        return
    end
    if nargin==2
        style = 'r*';
    end
    if isempty(M)
        if length(style)==3
            plot3(points(:,1),points(:,2),points(:,3),'*','Color',style)
        else
            plot3(points(:,1),points(:,2),points(:,3),style)
        end
    else
        if length(style)==3
            plot3(M.VERT(points,1),M.VERT(points,2),M.VERT(points,3),'*','Color',style)
        else
            plot3(M.VERT(points,1),M.VERT(points,2),M.VERT(points,3),style)
        end
    end
    axis equal
    rotate3d on
end
