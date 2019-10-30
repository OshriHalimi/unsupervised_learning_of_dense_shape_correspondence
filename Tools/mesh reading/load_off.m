function shape = load_off(filename)

shape = [];

f = fopen(filename, 'rt');

n = '';
while isempty(n)
    fgetl(f);
    n = sscanf(fgetl(f), '%d %d %d');
end

nv = n(1);
nt = n(2);
data = fscanf(f, '%f');

if length(data) == nv*3+nt*4
    data(3*nv+1:4:end)=[];
elseif length(data) ~= nv*3+nt*3
    fclose(f);
    error('load_off(): The mesh seems to be composed of non-triangular faces.');
end

shape.TRIV = reshape(data(end-3*nt+1:end), [3 nt])';
data = data(1:end-3*nt);
data = reshape(data, [length(data)/nv nv]);
shape.VERT = data';

fclose(f);

if min(min(shape.TRIV))==0
    shape.TRIV = 1+shape.TRIV;
end

shape.n = size(shape.VERT,1);
shape.m = size(shape.TRIV,1);

end