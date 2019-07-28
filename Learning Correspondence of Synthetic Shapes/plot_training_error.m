load('.\Results\train_faust_synthetic\training_error.mat')

title('Unsupervised Training Process')
yyaxis left
semilogy(error_vec_unsupervised);

yyaxis right
semilogy(error_vec_supervised);

xlabel('Number of mini-batches')