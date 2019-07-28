import time
import tensorflow as tf
import scipy.io as sio
import os

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_evecs', 120,
					 'number of eigenvectors used for representation. The first 500 are precomputed and stored in input')
flags.DEFINE_string('test_shapes_dir', './faust_synthetic/network_data/', '')
flags.DEFINE_string('log_dir', './Results/train_faust_synthetic', 'directory to save models and results')


def get_test_pair(part_fname, model_fname):
	input_data = {}
	# load model, part and labels
	part_file = '%s/%s'%(FLAGS.test_shapes_dir, part_fname)
	model_file = '%s/%s'%(FLAGS.test_shapes_dir, model_fname)
        
	input_data.update(sio.loadmat(part_file))  # this loads the part but with a model name so next line re-names
	input_data['part_evecs'] = input_data['model_evecs']
	del input_data['model_evecs']
	input_data['part_evecs_trans'] = input_data['model_evecs_trans']
	del input_data['model_evecs_trans']
	input_data['part_shot'] = input_data['model_shot']
	del input_data['model_shot']
		
	input_data.update(sio.loadmat(model_file))
	
	return input_data
	
def get_test_list():
	test_pairs = []
	f = open('./test_pairs.txt',mode='r', encoding='utf-8-sig')
	for line in f:
		test_pairs.append(line.split())
	f.close()

	return test_pairs
  

def run_test():
	# start session
	sess = tf.Session()

	print('restoring graph...')
	saver = tf.train.import_meta_graph('%s/model_unsupervised.ckpt-100.meta'%FLAGS.log_dir)
	saver.restore(sess, tf.train.latest_checkpoint('%s'%FLAGS.log_dir))
	graph = tf.get_default_graph()


	# retrieve placeholder variables
	part_shot = graph.get_tensor_by_name('part_shot:0')
	model_shot = graph.get_tensor_by_name('model_shot:0')
	dist_map = graph.get_tensor_by_name('model_dist_map:0')
	part_evecs = graph.get_tensor_by_name('part_evecs:0')
	part_evecs_trans = graph.get_tensor_by_name('part_evecs_trans:0')
	model_evecs = graph.get_tensor_by_name('model_evecs:0')
	model_evecs_trans = graph.get_tensor_by_name('model_evecs_trans:0')
	phase = graph.get_tensor_by_name('phase:0')

	# retrieve variables to run
	Ct_est = graph.get_tensor_by_name('matrix_solve_ls/cholesky_solve/MatrixTriangularSolve:0') #OH: Bug fixed, see: https://github.com/orlitany/DeepFunctionalMaps/issues/1
	softCorr = graph.get_tensor_by_name('pointwise_corr_loss/soft_correspondences:0')

	# read list of pairs to test on
	test_list = get_test_list()

	for test_pair in test_list:
		input_data = get_test_pair(test_pair[0], test_pair[1])

		feed_dict = {phase: True,
					 part_shot: [input_data['part_shot']],
					 model_shot: [input_data['model_shot']],
					 dist_map: [[[None]]],
					 part_evecs: [input_data['part_evecs'][:, 0:FLAGS.num_evecs]],
					 part_evecs_trans: [input_data['part_evecs_trans'][0:FLAGS.num_evecs, :]],
					 model_evecs: [input_data['model_evecs'][:, 0:FLAGS.num_evecs]],
					 model_evecs_trans: [input_data['model_evecs_trans'][0:FLAGS.num_evecs, :]],
					 }

		t = time.time()
		Ct_est_, softCorr_  = sess.run([Ct_est, softCorr], feed_dict=feed_dict)
		print('Computed correspondences for pair: %s, %s. Took %f seconds', test_pair[0], test_pair[1], time.time() - t)

		params_to_save = {}
		params_to_save['C_est'] = Ct_est_.transpose([0, 2, 1])
		params_to_save['softCorr'] = softCorr_

		if not os.path.isdir('./Results/test_faust_synthetic/'):
			os.mkdir('./Results/test_faust_synthetic/')

		sio.savemat('./Results/test_faust_synthetic/' + '{}_{}.mat'.format(test_pair[0][7:10], test_pair[1][7:10]), params_to_save)


def main(_):
  run_test()

if __name__ == '__main__':
  tf.app.run()    
