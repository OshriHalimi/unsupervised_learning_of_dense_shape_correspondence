function [x, fobj] = find_opt_alpha(opts)
% Copyright (c) 2015 Anastasia Dubrovina and Yonathan Aflalo

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%      Solve least squares optimization problem
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


USE_ALL_ALPHA = 0; % Specify whether you want to add constraints on alpha (0) or not (1)

if (USE_ALL_ALPHA)
    alpha1 = opts.alpha1;
    alpha2 = opts.alpha2;
    delta1 = opts.F1;
    delta2 = opts.F2;
else
    alpha1 = opts.alpha1(2:end,2:end);
    alpha2 = opts.alpha2(2:end,2:end);
    delta1 = opts.F1(2:end,:);
    delta2 = opts.F2(2:end,:);
end

mu1=opts.mu1;
mu2=opts.mu2;

Nvec_basis = size(alpha1,1);
Nf = size(delta1, 2);

I = speye(Nvec_basis);
P_delta = vecperm(Nvec_basis, Nf);

M_alpha = kron(alpha1', I) - kron(I, alpha2);
M_delta1 = kron(delta1', I);
M_delta2 = P_delta*kron(I, delta2');
b_delta1 = delta1(:);
b_delta2 = delta2(:);

% Use all alpha
if (USE_ALL_ALPHA)
    A = (mu1*(M_alpha'*M_alpha) + mu2*(M_delta1'*M_delta1) + mu2*(M_delta2'*M_delta2));
    b = mu2*M_delta1'*b_delta2 + mu2*M_delta2'*b_delta1;
else
    M_alpha1 = kron(opts.alpha1(2:end, 1)', I);
    M_alpha2 = kron(I, opts.alpha2(1,2:end));
    b_alpha1 = opts.alpha1(1,2:end)';
    b_alpha2 = opts.alpha2(2:end,1);
    %
    A = (mu1*(M_alpha'*M_alpha) + mu1*(M_alpha1'*M_alpha1) + mu1*(M_alpha2'*M_alpha2) + mu2*(M_delta1'*M_delta1) + mu2*(M_delta2'*M_delta2));
    b = mu1*M_alpha2'*b_alpha1 + mu1*M_alpha1'*b_alpha2 + mu2*M_delta1'*b_delta2 + mu2*M_delta2'*b_delta1;
end

% 
x = A \ b;

fobj = norm(A*x - b);

end