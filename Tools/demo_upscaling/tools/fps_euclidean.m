function [ S ] = fps_euclidean(V, n, seed)

S = zeros(n,1);
S(1) = seed;
d = pdist2(V,V(seed,:));

for i=2:n
    [~,m] = max(d);
    S(i) = m(1);
    d = min(pdist2(V,V(S(i),:)) , d);
end

end
