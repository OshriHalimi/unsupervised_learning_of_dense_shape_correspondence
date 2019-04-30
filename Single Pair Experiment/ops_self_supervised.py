import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS


def pointwise_corr_layer(C_est, model_evecs, part_evecs_trans, model_dist_map, part_dist_map):
    """Point-wise correlation between learned descriptors.

    Args:
        C_est: estimated C matrix from previous layer.
        model_evecs: eigen vectors of model shape.
        part_evecs_trans: eigen vectors of part shape, transposed with area preservation factor.
        dist_map: matrix of geodesic distances on model.

    """
    P = tf.matmul(tf.matmul(model_evecs, C_est), part_evecs_trans)
    P = tf.abs(P)

    P_norm = tf.nn.l2_normalize(P, dim=1, name='soft_correspondences')  # normalize the columns #OH: dim 0 is the batch dimension
    #Diffusion_part_norm = tf.nn.l2_normalize(tf.math.exp(-part_dist_map/200), dim=1,  name='diffusion_part')
    #Diffusion_model_norm = tf.nn.l2_normalize(tf.math.exp(-model_dist_map/200), dim=1, name='diffusion_model')

    #unsupervised loss calculation
    avg_distance_on_model_after_map = tf.einsum('kmn,kmi,knj->kij', model_dist_map, tf.pow(P_norm,2), tf.pow(P_norm,2)) #OH: k is the batch dimension i,j is the vertices on the part, m,n are the vertices on the model
    avg_distortion_after_map = avg_distance_on_model_after_map - part_dist_map
    unsupervised_loss = tf.nn.l2_loss(avg_distortion_after_map)
    unsupervised_loss /= tf.to_float(tf.shape(P)[0] * tf.shape(P)[2] * tf.shape(P)[2]) #OH:normalize by the batch size multiplied by the number of vertex-pairs on the part model. This measures the average distortion

    #OH: matmul support batch multiplication, 2 last dimensions are multiplied for every batch sample
    #topological_loss = tf.nn.l2_loss(tf.matmul(Diffusion_model_norm,tf.pow(P_norm,2)) - tf.matmul(tf.pow(P_norm,2),Diffusion_part_norm)) #OH: each columns, for each vertex on the part, is the deviation between two distributions on the model
    #topological_loss /= tf.to_float(tf.shape(P)[0] * tf.shape(P)[1]) #OH: normalize by the batch size and the number of vertices on the part
    #unsupervised_loss += 100*topological_loss
    return P_norm, unsupervised_loss


def res_layer(x_in, dims_out, scope, phase):
    """A residual layer implementation.

    Args:
        x_in: input descriptor per point (dims = batch_size X #pts X #channels)
        dims_out: num channles in output. Usually the same as input for a standard resnet layer.
        scope: scope name for variable sharing.
        phase: train\test.

    """
    with tf.variable_scope(scope):
        x = tf.contrib.layers.fully_connected(x_in, dims_out, activation_fn=None, scope='dense_1')
        x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase, scope='bn_1')
        x = tf.nn.relu(x, 'relu')
        x = tf.contrib.layers.fully_connected(x, dims_out, activation_fn=None, scope='dense_2')
        x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase, scope='bn_2')

        # if dims_out change, modify input via linear projection (as suggested in resNet)
        if not x_in.get_shape().as_list()[-1] == dims_out:
            x_in = tf.contrib.layers.fully_connected(x_in, dims_out, activation_fn=None, scope='projection')

        x += x_in

        return tf.nn.relu(x)


def solve_ls(A, B):
    """functional maps layer.

    Args:
        A: part descriptors projected onto part shape eigenvectors.
        B: model descriptors projected onto model shape eigenvectors.

    Returns:
        Ct_est: estimated C (transposed), such that CA ~= B
        safeguard_inverse:
    """

    # transpose input matrices
    At = tf.transpose(A, [0, 2, 1])
    Bt = tf.transpose(B, [0, 2, 1])

    # solve C via least-squares
    Ct_est = tf.matrix_solve_ls(At, Bt)
    C_est = tf.transpose(Ct_est, [0, 2, 1])

    # calculate error for safeguarding
    safeguard_inverse = tf.nn.l2_loss(tf.matmul(At,Ct_est) - Bt) / tf.to_float(tf.reduce_prod(tf.shape(A)))

    return C_est, safeguard_inverse
