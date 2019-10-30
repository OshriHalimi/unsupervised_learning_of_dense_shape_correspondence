function shape = load_ply(filename)

[tri,vert] = ply_read(filename, 'tri');
shape = {};
shape.VERT = vert';
shape.TRIV = tri';

shape.n = size(shape.VERT,1);
shape.m = size(shape.TRIV,1);

end