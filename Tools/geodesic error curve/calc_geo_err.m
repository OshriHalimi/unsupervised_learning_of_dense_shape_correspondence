function errs = calc_geo_err(matches, gt_matches, D)

nm = length(matches);
errs = zeros(nm,1);

for i=1:nm
    errs(i) = D( matches(i), gt_matches(i) );
end

errs = errs ./ max(max(D));

end
