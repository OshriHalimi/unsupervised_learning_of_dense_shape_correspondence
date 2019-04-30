import os
import time
import tensorflow as tf
import numpy as np
import scipy.io as sio

from models_self_supervised import fmnet_model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

flags = tf.app.flags
FLAGS = flags.FLAGS

# training params
flags.DEFINE_float('learning_rate', 1e-4, 'initial learning rate.') #OH: originally was 1e-3
flags.DEFINE_integer('batch_size', 1, 'batch size.') #OH: originally was 32
flags.DEFINE_integer('queue_size', 10, '')

# architecture parameters
flags.DEFINE_integer('num_layers', 7, 'network depth')
flags.DEFINE_integer('num_evecs', 150,
					 'number of eigenvectors used for representation. The first 500 are precomputed and stored in input')
flags.DEFINE_integer('dim_shot', 352, '')
flags.DEFINE_integer('num_vertices_model', 5156,'')
flags.DEFINE_integer('num_vertices_part', 5156,'')
# data parameters
flags.DEFINE_string('models_dir', './tf_artist/', '')
flags.DEFINE_string('dist_maps', './tf_artist/', '')

flags.DEFINE_string('log_dir', './Results/artist_checkpoint', 'directory to save models and results')
flags.DEFINE_integer('max_train_iter', 400, '')
flags.DEFINE_integer('save_summaries_secs', 60, '')
flags.DEFINE_integer('save_model_secs', 1200, '')
flags.DEFINE_string('master', '', '')
flags.DEFINE_integer('run_validation_every', 150, '')
flags.DEFINE_integer('validation_size', 10, '')

# globals
dist_maps = {}
train_subjects = [0,1]
validation_subjects= []

def get_input_pair(batch_size=1):
	dataset = 'train'
	batch_input = {'part_evecs': np.zeros((batch_size, FLAGS.num_vertices_part, FLAGS.num_evecs)),
				   'model_evecs': np.zeros((batch_size, FLAGS.num_vertices_part, FLAGS.num_evecs)),
				   'part_evecs_trans': np.zeros((batch_size, FLAGS.num_evecs, FLAGS.num_vertices_part)),
				   'model_evecs_trans': np.zeros((batch_size, FLAGS.num_evecs, FLAGS.num_vertices_model)),
				   'part_shot': np.zeros((batch_size, FLAGS.num_vertices_model, FLAGS.dim_shot)),
				   'model_shot': np.zeros((batch_size, FLAGS.num_vertices_model, FLAGS.dim_shot))
				   }

	batch_model_dist = np.zeros((batch_size, FLAGS.num_vertices_model, FLAGS.num_vertices_model))
	batch_part_dist = np.zeros((batch_size, FLAGS.num_vertices_part, FLAGS.num_vertices_part))

	for i_batch in range(batch_size):
		i_model = 1
		i_part = 0

		batch_input_, batch_model_dist_, batch_part_dist_ = get_pair_from_ram(i_model, i_part, dataset)

		batch_input_['part_labels'] = range(np.shape(batch_input_['part_evecs'])[0])  # replace once we switch to scans
		batch_input_['model_labels'] = range(np.shape(batch_input_['model_evecs'])[0])

		#OH: instead of random permutation of joint_labels, that is common for both part and model - we randomize the subsampled labels separately for the part and for the model.
		#Since the training is unsupervised, we don't rely on the ground-truth correspondence to produce exactly-matching subsets of the vertcies.
		part_random_labels = np.random.permutation(batch_input_['part_labels'])[:FLAGS.num_vertices_part]
		ind_dict_part = {value: ind for ind, value in enumerate(batch_input_['part_labels'])}
		ind_part = [ind_dict_part[x] for x in part_random_labels]

		model_random_labels = np.random.permutation(batch_input_['model_labels'])[:FLAGS.num_vertices_model]
		ind_dict_model = {value: ind for ind, value in enumerate(batch_input_['model_labels'])}
		ind_model = [ind_dict_model[x] for x in model_random_labels]

		#assert len(ind_part) == len(ind_model), 'number of indices must be equal'

		batch_model_dist[i_batch] = batch_model_dist_[ind_model, :][:, ind_model]  # slice the subsampled indices
		batch_part_dist[i_batch] = batch_part_dist_[ind_part, :][:, ind_part]
		batch_input['part_evecs'][i_batch] = batch_input_['part_evecs'][ind_part, :]
		batch_input['part_evecs_trans'][i_batch] = batch_input_['part_evecs_trans'][:, ind_part]
		batch_input['part_shot'][i_batch] = batch_input_['part_shot'][ind_part, :]
		batch_input['model_evecs'][i_batch] = batch_input_['model_evecs'][ind_model, :]
		batch_input['model_evecs_trans'][i_batch] = batch_input_['model_evecs_trans'][:, ind_model]
		batch_input['model_shot'][i_batch] = batch_input_['model_shot'][ind_model, :]

	return batch_input, batch_model_dist, batch_part_dist


def get_pair_from_ram(i_model, i_part, dataset):
	input_data = {}

	if dataset == 'train':
		input_data['part_evecs'] = models_train[i_part]['model_evecs']
		input_data['part_evecs_trans'] = models_train[i_part]['model_evecs_trans']
		input_data['part_shot'] = models_train[i_part]['model_shot']
		input_data.update(models_train[i_model])
	else:
		input_data['part_evecs'] = models_val[i_part]['model_evecs']
		input_data['part_evecs_trans'] = models_val[i_part]['model_evecs_trans']
		input_data['part_shot'] = models_val[i_part]['model_shot']
		input_data.update(models_val[i_model])

	# m_star from dist_map
	m_star = dist_maps[i_model]
	p_star = dist_maps[i_part]

	return input_data, m_star, p_star


def load_models_to_ram():
	global models_train
	models_train = {}
	global models_val
	models_val = {}

	# load model, part and labels
	for i in range(train_subjects.__len__()):
		model_file = FLAGS.models_dir + '/' + 'model_%d.mat' % train_subjects[i]
		input_data = sio.loadmat(model_file)
		input_data['model_evecs'] = input_data['model_evecs'][:, 0:FLAGS.num_evecs]
		input_data['model_evecs_trans'] = input_data['model_evecs_trans'][0:FLAGS.num_evecs, :]
		models_train[i] = input_data

	#for i in validation_subjects:
	#	model_file = FLAGS.models_dir + '/' + '%d.mat' % i
	#	input_data = sio.loadmat(model_file)
	#	input_data['model_evecs'] = input_data['model_evecs'][:, 0:FLAGS.num_evecs]
	#	input_data['model_evecs_trans'] = input_data['model_evecs_trans'][0:FLAGS.num_evecs, :]
	#	models_val[i] = input_data


def load_dist_maps():
	print('loading dist maps...')
	# load distance maps to memory for both training and validation sets
	for i in range(train_subjects.__len__()): #+ validation_subjects:
		global dist_maps
		d = sio.loadmat(FLAGS.dist_maps + 'model_%d_dist.mat' % train_subjects[i])
		dist_maps[i] = d['D']


def run_training():

	print('log_dir=%s' % FLAGS.log_dir)
	if not os.path.isdir(FLAGS.log_dir):
		os.makedirs(FLAGS.log_dir)  # changed from mkdir
	print('num_evecs=%d' % FLAGS.num_evecs)

	print('building graph...')
	with tf.Graph().as_default():

		# Set placeholders for inputs
		part_shot = tf.placeholder(tf.float32, shape=(None, None, FLAGS.dim_shot), name='part_shot')
		model_shot = tf.placeholder(tf.float32, shape=(None, None, FLAGS.dim_shot), name='model_shot')
		model_dist_map = tf.placeholder(tf.float32, shape=(None, None, None), name='model_dist_map')
		part_dist_map = tf.placeholder(tf.float32, shape=(None, None, None), name='part_dist_map')
		part2model_ind_gt = tf.placeholder(tf.float32, shape=(None, None), name='part2model_groundtruth')
		part_evecs = tf.placeholder(tf.float32, shape= (None, None, FLAGS.num_evecs), name='part_evecs')
		part_evecs_trans = tf.placeholder(tf.float32, shape=(None, FLAGS.num_evecs, None), name='part_evecs_trans')
		model_evecs = tf.placeholder(tf.float32, shape= (None, None, FLAGS.num_evecs), name='model_evecs')
		model_evecs_trans = tf.placeholder(tf.float32, shape=(None, FLAGS.num_evecs, None), name='model_evecs_trans')

		# train\test switch flag
		phase = tf.placeholder(dtype=tf.bool, name='phase')

		unsupervised_loss, safeguard_inverse, P_norm, net = fmnet_model(phase, part_shot, model_shot,
																					  part_dist_map , model_dist_map,
																					  part_evecs, part_evecs_trans, model_evecs, model_evecs_trans)
		summary = tf.summary.scalar("num_evecs", float(FLAGS.num_evecs))

		global_step = tf.Variable(0, name='global_step', trainable=False)

		optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

		train_op = optimizer.minimize(unsupervised_loss, global_step=global_step)

		saver = tf.train.Saver(max_to_keep=100)
		sv = tf.train.Supervisor(logdir=FLAGS.log_dir,
								 init_op=tf.global_variables_initializer(),
								 local_init_op=tf.local_variables_initializer(),
								 global_step=global_step,
								 save_summaries_secs=FLAGS.save_summaries_secs,
								 save_model_secs=FLAGS.save_model_secs,
								 summary_op=None,
								 saver=saver)

		writer = sv.summary_writer


		print('starting session...')
		iteration = 0
		with sv.managed_session(master=FLAGS.master) as sess:

			print('loading data to ram...')
			load_models_to_ram()

			load_dist_maps()

			print('starting training loop...')
			while not sv.should_stop() and iteration < FLAGS.max_train_iter:
				iteration += 1
				start_time = time.time()

				input_data, mstar, pstar = get_input_pair(FLAGS.batch_size)

				feed_dict = {phase: True,
							 part_shot: input_data['part_shot'],
							 model_shot: input_data['model_shot'],
							 model_dist_map: mstar,
							 part_dist_map: pstar,
							 part_evecs: input_data['part_evecs'],
							 part_evecs_trans: input_data['part_evecs_trans'],
							 model_evecs: input_data['model_evecs'],
							 model_evecs_trans: input_data['model_evecs_trans'],
							 }

				step, my_unsupervised_loss, safeguard, _ = sess.run(
					[global_step, unsupervised_loss, safeguard_inverse, train_op], feed_dict=feed_dict)

				duration = time.time() - start_time

				print('train - step %d: unsupervised loss = %.4f  (%.3f sec)' % (step, my_unsupervised_loss, duration))


			saver.save(sess, FLAGS.log_dir + '/model.ckpt', global_step=step)
			writer.flush()
			sv.request_stop()
			sv.stop()


def main(_):
	run_training()


if __name__ == '__main__':
	tf.app.run()
