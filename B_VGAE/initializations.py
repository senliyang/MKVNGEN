import tensorflow.compat.v1 as tf
import numpy as np
np.random.seed(2024)
tf.random.set_random_seed(2024)
def weight_variable_glorot(input_dim, output_dim, name=""):
    tf.random.set_random_seed(2024)
    np.random.seed(2024)
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)
