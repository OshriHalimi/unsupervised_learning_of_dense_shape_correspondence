% Faust Synthetic
test_idx=[80:99];
pairs_list = [];
for i=1:20
    for j = 1:20
        pairs_list = [pairs_list; test_idx(i),test_idx(j)];
    end
end

% delete_idx = floor(pairs_list(:,1)/10) == floor(pairs_list(:,2)/10);
% pairs_list(delete_idx,:)=[];

fileID = fopen('test_pairs.txt','w');
for i=1:size(pairs_list,1)
    id1 = sprintf('%03d', pairs_list(i,1));
    id2 = sprintf('%03d', pairs_list(i,2));
    fprintf(fileID,['tr_reg_', id1, '.mat tr_reg_' ,id2, '.mat\n']);
end
fclose(fileID)
