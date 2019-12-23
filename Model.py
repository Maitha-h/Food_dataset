import tensorflow as tf

tf1 = tf.compat.v1


def batch_norm(op_name, inputs, is_training, decay=0.9997, epsilon=0.001, variable_getter=None):
    moving_collections = [tf1.GraphKeys.GLOBAL_VARIABLES, tf1.GraphKeys.MOVING_AVERAGE_VARIABLES]
    inputs_shape = inputs.getshape()
    params_shape = inputs_shape[-1:]

    with tf.device("/device:CPU:0"), tf1.variable_scope("vars/bns", None, [inputs],reuse=tf1.AUTO_REUSE):
        beta = tf1.get_variable("beta_"+op_name, shape=params_shape, initializer=tf.zeros_initializer(), custom_getter=variable_getter)
        moving_mean = tf1.get_variable("moving_mean_"+op_name, params_shape, initializer=tf.ones_initializer(), trainable=False, collection=moving_collections)
        moving_variance = tf1.get_variable("moving_variance_" + op_name, params_shape, initializer=tf.ones_initializer(), trainable=False, collections=moving_collections)


def training_func():
    return 0