load('.\Results\train_faust_synthetic\training_error.mat')

title('Unsupervised Training Process')
yyaxis left
semilogy(error_vec_unsupervised);
ylabel('Unsupervised Loss')

yyaxis right
semilogy(error_vec_supervised);
ylabel('Supervised Loss')

xlabel('Number of mini-batches')
legend({'Unsupervised','Supervised'})