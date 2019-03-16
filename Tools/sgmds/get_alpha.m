function Alpha = get_alpha(D, sample, Phi, lambda, miu) 
nv = size(Phi, 1); 
D = 0.5*(D + D');
% I = eye(N);
I = spdiags(ones(nv,1),0,nv,nv);
B = I(sample, :);
B_Phi = B * Phi;
M1 = (lambda.^2 + miu * (B_Phi')*B_Phi); %lambda for gradient. lambda.^2 for laplacian
M2 = miu*B_Phi';
M = M1\M2;

Alpha = M*D*M';

