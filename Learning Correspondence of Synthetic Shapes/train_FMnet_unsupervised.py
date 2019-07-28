import os
import time
import tensorflow as tf
import numpy as np
import scipy.io as sio

from models import fmnet_model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

flags = tf.app.flags
FLAGS = flags.FLAGS

# training params
flags.DEFINE_float('learning_rate', 1e-3, 'initial learning rate.') #OH: originally was 1e-3
flags.DEFINE_integer('batch_size', 4, 'batch size.') #OH: originally was 32
flags.DEFINE_integer('queue_size', 10, '')

# architecture parameters
flags.DEFINE_integer('num_layers', 7, 'network depth')
flags.DEFINE_integer('num_evecs', 120,
					 'number of eigenvectors used for representation. The first 500 are precomputed and stored in input')
flags.DEFINE_integer('dim_shot', 352, '')
flags.DEFINE_integer('num_vertices', 6890,'')
# data parameters
flags.DEFINE_string('models_dir', './faust_synthetic/network_data/', '')
flags.DEFINE_string('dist_maps', './faust_synthetic/distance_matrix/', '')

flags.DEFINE_string('log_dir', './Results/train_faust_synthetic', 'directory to save models and results')
flags.DEFINE_integer('max_train_iter', 3000, '')
flags.DEFINE_integer('save_summaries_secs', 60, '')
flags.DEFINE_integer('save_model_secs', 1200, '')
flags.DEFINE_string('master', '', '')

# globals
error_vec_unsupervised = []
error_vec_supervised = []
train_subjects = [0, 1, 2, 3, 4, 5, 6, 7]

flags.DEFINE_integer('num_poses_per_subject_total', 10, '')
dist_maps = {}


def get_input_pair(batch_size=1, num_vertices=FLAGS.num_vertices):
	dataset = 'train'
	batch_input = {'part_evecs': np.zeros((batch_size, num_vertices, FLAGS.num_evecs)),
				   'model_evecs': np.zeros((batch_size, num_vertices, FLAGS.num_evecs)),
				   'part_evecs_trans': np.zeros((batch_size, FLAGS.num_evecs, num_vertices)),
				   'model_evecs_trans': np.zeros((batch_size, FLAGS.num_evecs, num_vertices)),
				   'part_shot': np.zeros((batch_size, num_vertices, FLAGS.dim_shot)),
				   'model_shot': np.zeros((batch_size, num_vertices, FLAGS.dim_shot))
				   }

	batch_model_dist = np.zeros((batch_size, num_vertices, num_vertices))
	batch_part_dist = np.zeros((batch_size, num_vertices, num_vertices))
	batch_part_ind2model_ind = np.zeros((batch_size,num_vertices))
	for i_batch in range(batch_size):
		i_subject1 = np.random.choice(train_subjects)  # model #OH: randomize model subject index [0...7], for train
		i_subject2 = np.random.choice(train_subjects) #OH: randomize part subject index [0...7], for train
		i_model = FLAGS.num_poses_per_subject_total * i_subject1 + \
				  np.random.randint(0, FLAGS.num_poses_per_subject_total, 1)[0] #OH: numpy.random.randint(low (inclusive), high(exclusize), size).
																				#OH: Randomize model index [0...79], for train
		i_part = FLAGS.num_poses_per_subject_total * i_subject2 + \
				 np.random.randint(0, FLAGS.num_poses_per_subject_total, 1)[0] #OH: Randomize part index [0...79], for train

		batch_input_, batch_model_dist_, batch_part_dist_ = get_pair_from_ram(i_subject1, i_subject2, i_model, i_part, dataset)

		batch_input_['part_labels'] = range(np.shape(batch_input_['part_evecs'])[0])  # replace once we switch to scans
		batch_input_['model_labels'] = range(np.shape(batch_input_['model_evecs'])[0])

		#OH: instead of random permutation of joint_labels, that is common for both part and model - we randomize the subsampled labels separately for the part and for the model.
		#Since the training is unsupervised, we don't rely on the ground-truth correspondence to produce exactly-matching subsets of the vertcies.
		part_random_labels = np.random.permutation(batch_input_['part_labels'])[:num_vertices]
		ind_dict_part = {value: ind for ind, value in enumerate(batch_input_['part_labels'])}
		ind_part = [ind_dict_part[x] for x in part_random_labels]

		model_random_labels = np.random.permutation(batch_input_['model_labels'])[:num_vertices]
		model_label2ind_dict = {label: ind for ind, label in enumerate(model_random_labels)}
		ind_dict_model = {value: ind for ind, value in enumerate(batch_input_['model_labels'])}
		ind_model = [ind_dict_model[x] for x in model_random_labels]

		assert len(ind_part) == len(ind_model), 'number of indices must be equal'

		# OH: This array "batch_part_ind2model_ind" is used only for the task of monitoring the supervised loss during the unsupervised training process.
		# The training is completely independent of the ground truth knowledge
		batch_part_ind2model_ind[i_batch] = np.array([model_label2ind_dict[part_label] for part_label in part_random_labels])
		batch_model_dist[i_batch] = batch_model_dist_[ind_model, :][:, ind_model]  # slice the subsampled indices
		batch_part_dist[i_batch] = batch_part_dist_[ind_part, :][:, ind_part]
		batch_input['part_evecs'][i_batch] = batch_input_['part_evecs'][ind_part, :]
		batch_input['part_evecs_trans'][i_batch] = batch_input_['part_evecs_trans'][:, ind_part]
		batch_input['part_shot'][i_batch] = batch_input_['part_shot'][ind_part, :]
		batch_input['model_evecs'][i_batch] = batch_input_['model_evecs'][ind_model, :]
		batch_input['model_evecs_trans'][i_batch] = batch_input_['model_evecs_trans'][:, ind_model]
		batch_input['model_shot'][i_batch] = batch_input_['model_shot'][ind_model, :]

	return batch_input, batch_model_dist, batch_part_dist, batch_part_ind2model_ind


def get_pair_from_ram(i_subject_model, i_subject_part, i_model, i_part, dataset):
	input_data = {}

	if dataset == 'train':
		input_data['part_evecs'] = models_train[i_part]['model_evecs']
		input_data['part_evecs_trans'] = models_train[i_part]['model_evecs_trans']
		input_data['part_shot'] = models_train[i_part]['model_shot']
		input_data.update(models_train[i_model])

	# m_star from dist_map
	m_star = dist_maps[i_subject_model]
	p_star = dist_maps[i_subject_part]

	return input_data, m_star, p_star


def load_models_to_ram():
	global models_train
	models_train = {}

	# load model and part
	for i_subject in train_subjects:
		for i_model in range(i_subject * FLAGS.num_poses_per_subject_total,
							 FLAGS.num_poses_per_subject_total * (i_subject + 1)):
			model_file = FLAGS.models_dir + 'tr_reg_%.3d.mat' % i_model
			input_data = sio.loadmat(model_file)
			input_data['model_evecs'] = input_data['model_evecs'][:, 0:FLAGS.num_evecs]
			input_data['model_evecs_trans'] = input_data['model_evecs_trans'][0:FLAGS.num_evecs, :]
			models_train[i_model] = input_data

def load_dist_maps():
	print('loading dist maps...')
	# load distance maps to memory for training set
	for i_subject in train_subjects:
		global dist_maps
		d = sio.loadmat(FLAGS.dist_maps + 'tr_reg_%.3d.mat' % (i_subject * FLAGS.num_poses_per_subject_total))
		dist_maps[i_subject] = d['D']


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

		net_loss, unsupervised_loss, safeguard_inverse, merged, P_norm, net = fmnet_model(phase, part_shot, model_shot,
																					  part_dist_map , model_dist_map, part2model_ind_gt,
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

				input_data, mstar, pstar, p2m_ind_gt = get_input_pair(FLAGS.batch_size)

				feed_dict = {phase: True,
							 part_shot: input_data['part_shot'],
							 model_shot: input_data['model_shot'],
							 model_dist_map: mstar,
							 part_dist_map: pstar,
							 part2model_ind_gt: p2m_ind_gt,
							 part_evecs: input_data['part_evecs'],
							 part_evecs_trans: input_data['part_evecs_trans'],
							 model_evecs: input_data['model_evecs'],
							 model_evecs_trans: input_data['model_evecs_trans'],
							 }

				summaries, step, my_loss, my_unsupervised_loss, safeguard, _ = sess.run(
					[merged, global_step, net_loss, unsupervised_loss, safeguard_inverse, train_op], feed_dict=feed_dict)
				writer.add_summary(summaries, step)
				summary_ = sess.run(summary)
				writer.add_summary(summary_, step)

				duration = time.time() - start_time

				print('train - step %d: loss = %.4f unsupervised loss = %.4f(%.3f sec)' % (step, my_loss, my_unsupervised_loss, duration))
				error_vec_unsupervised.append(my_unsupervised_loss)
				error_vec_supervised.append(my_loss)


			saver.save(sess, FLAGS.log_dir + '/model_unsupervised.ckpt', global_step=step)
			writer.flush()
			sv.request_stop()
			sv.stop()

	#OH: save training error
	params_to_save = {}
	params_to_save['error_vec_unsupervised'] = np.array(error_vec_unsupervised)
	params_to_save['error_vec_supervised'] = np.array(error_vec_supervised)
	sio.savemat(FLAGS.log_dir  + '/training_error.mat', params_to_save)

	# OH: plot training error
	hu = plt.plot(np.array(error_vec_unsupervised),'r')
	hs = plt.plot(np.array(error_vec_supervised),'b')
	red_patch = mpatches.Patch(color='red', label='Unsupervised')
	blue_patch = mpatches.Patch(color='blue', label='Supervised')
	plt.legend(handles=[red_patch,blue_patch])
	plt.title('Training process with the unsupervised loss')
	plt.xlabel('Training step')
	plt.ylabel('Loss')
	plt.show()


def main(_):
	run_training()


if __name__ == '__main__':
	tf.app.run()
