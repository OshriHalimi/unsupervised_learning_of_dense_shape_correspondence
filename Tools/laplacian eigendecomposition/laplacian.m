function [ M,S ] = laplacian( V, F )

M = massmatrix(V,F);
S = stiffnessmatrix(V,F);

end

function [ M ] = massmatrix( V, F )
%MASSMATRIX generates an n-by-n mass matrix.
%  Input:
%   - V : n-by-3 matrix contains the 3D coordinates of the n vertices in
%         each of it's rows.
%   - F : m-by-3 matrix contains the indeces of the triangles' vertices.
%
%  Output:
%   - M : n-by-n mass matrix contains the values as described in the
%         lecture.
if  (size(V,2)~=3)
    V = V';
end
assert(size(V,2)==3);
if  (size(F,2)~=3)
    F = F';
end
assert(size(F,2)==3);
m = size(F,1);
n = size(V,1);

%triangle areas
A = facearea(V, F);

indecesI = [F(:,1);F(:,2);F(:,3);F(:,3);F(:,2);F(:,1)];
indecesJ = [F(:,2);F(:,3);F(:,1);F(:,2);F(:,1);F(:,3)];
values   = [A(:)  ;A(:)  ;A(:)  ;A(:)  ;A(:)  ;A(:)  ]*(1/12);
M = sparse(indecesI, indecesJ, values,n,n);
M = M+sparse(1:n,1:n,sum(M));

end

function [ S ] = stiffnessmatrix( V, F )
%COTANMATRIX_NEW Summary of this function goes here
%   Detailed explanation goes here


if  (size(V,2)~=3)
    V = V';
end
assert(size(V,2)==3);
n = length(V);
if  (size(F,2)~=3)
    F = F';
end
assert(size(F,2)==3);

%% Compute matrix containing the cotangens-values
vm = V(F(:,1),:);
vn = V(F(:,2),:);
vk = V(F(:,3),:);

size(vm);
% calculate edge lengths
vmn = sqrt(sum((vm-vn).^2,2));
vnk = sqrt(sum((vn-vk).^2,2));
vkm = sqrt(sum((vk-vm).^2,2));

cosam = (vmn.^2.+vkm.^2-vnk.^2)./(2*vmn.*vkm);
cosan = (vnk.^2+vmn.^2.-vkm.^2)./(2*vnk.*vmn);
cosak = (vkm.^2+vnk.^2.-vmn.^2)./(2*vkm.*vnk);

cotam = cosam ./ sqrt(1-cosam.^2);
cotan = cosan ./ sqrt(1-cosan.^2);
cotak = cosak ./ sqrt(1-cosak.^2);

A = 0.5*[cotam cotan cotak];

i = [    F(:,1);    F(:,1);  F(:,1);  F(:,2);      F(:,2);    F(:,2);  F(:,3);  F(:,3);      F(:,3)];
j = [    F(:,1);    F(:,2);  F(:,3);  F(:,1);      F(:,2);    F(:,3);  F(:,1);  F(:,2);      F(:,3)];
s = [A(:,3)+A(:,2); -A(:,3); -A(:,2); -A(:,3); A(:,3)+A(:,1); -A(:,1); -A(:,2); -A(:,1); A(:,1)+A(:,2)];

S = sparse(i,j,s,n,n);


end

function [ f ] = facearea( V, F )
%FACEAREA Computes the area of each face (triangle) of mesh M.
%   Returns vector f where f(i)==area of triangle i.


V1 = V(F(:,2),:)-V(F(:,1),:);
V2 = V(F(:,3),:)-V(F(:,1),:);
f = cross(V1,V2,2);
f = sqrt(sum(f.^2,2)).*0.5;
end

