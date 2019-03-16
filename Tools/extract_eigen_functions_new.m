function [Phi,S,Lambda,M] = extract_eigen_functions_new(shape,k)
    
    [M,S]=laplacian([shape.X shape.Y shape.Z],shape.TRIV);
    M = diag(sum(M,2));
    [Phi,Lambda] = eigs(-S,M,k,1e-5);
    Lambda = diag(Lambda);
    [Lambda,idx] = sort(Lambda,'descend');
    Phi = Phi(:,idx);
%     PhiI = Phi'*M;
    Lambda = abs(Lambda); %added this to return positive eigen values. 

end
