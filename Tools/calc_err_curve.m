function curve = calc_err_curve(errors, thresholds)

curve = zeros(1, length(thresholds));
for i=1:length(thresholds)
    curve(i) = 100*sum(errors <= thresholds(i)) / length(errors);
end

end
