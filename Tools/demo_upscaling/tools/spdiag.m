function D = spdiag(d)
n = length(d);
D = sparse(1:n, 1:n, d);
end
