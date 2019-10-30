function [M, is_outlier] = cleanup(S)
    
missing = setdiff(1:size(S.VERT,1), unique(S.TRIV(:)));
is_outlier = false(size(S.VERT,1),1);
is_outlier(missing) = true;
M = removeVertices(S, is_outlier, false);

fprintf('%d outliers detected.\n', sum(is_outlier));

end

function N = removeVertices(N,v,apply_to_gt)
    tri2keep = sum(v(N.TRIV),2)==0;
    N.TRIV = N.TRIV(tri2keep,:);
    N.VERT = N.VERT(~v,:);
    reindex(~v)=1:sum(~v);
    N.TRIV = reindex(N.TRIV); 
    if (nargin==2) || (nargin==3 && apply_to_gt)
        N.gt = N.gt(~v);
    end
    N.m = size(N.TRIV,1);
    N.n = size(N.VERT,1);
end

