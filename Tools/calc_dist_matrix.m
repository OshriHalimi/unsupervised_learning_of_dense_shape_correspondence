function D = calc_dist_matrix(M, samples)

if nargin==1 || isempty(samples)
    samples = 1:M.n;
end

march = fastmarchmex('init', int32(M.TRIV-1), double(M.VERT(:,1)), double(M.VERT(:,2)), double(M.VERT(:,3)));

D = zeros(length(samples));

for i=1:length(samples)
%     fprintf('(%d/%d)\n', i, length(samples));
    source = inf(M.n,1);
    source(samples(i)) = 0;
    d = fastmarchmex('march', march, double(source));
    D(:,i) = d(samples);
end

fastmarchmex('deinit', march);

D = 0.5*(D+D');

end
