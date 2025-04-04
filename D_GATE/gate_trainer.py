import tensorflow._api.v2.compat.v1 as tf
from gate_method import GATE
import data_preprocess
tf.disable_eager_execution()
tf.random.set_random_seed(2024)
class GATETrainer():
    tf.random.set_random_seed(2024)

    def __init__(self, args):
        tf.random.set_random_seed(2024)

        self.args = args
        self.build_placeholders()
        gate = GATE(args.hidden_dims, args.lambda_)
        self.loss, self.H, self.C = gate(self.A, self.X, self.R, self.S)
        self.optimize(self.loss)
        self.build_session()

    def build_placeholders(self):
        tf.random.set_random_seed(2024)

        self.A = tf.sparse_placeholder(dtype=tf.float32)
        self.X = tf.placeholder(dtype=tf.float32)
        self.S = tf.placeholder(tf.int64)
        self.R = tf.placeholder(tf.int64)

    def build_session(self, gpu= True):
        tf.random.set_random_seed(2024)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if gpu == False:
            config.intra_op_parallelism_threads = 0
            config.inter_op_parallelism_threads = 0
        self.session = tf.Session(config=config)
        self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])


    def optimize(self, loss):
        tf.random.set_random_seed(2024)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.args.gradient_clipping)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))


    def __call__(self, A, X, S, R):
        tf.random.set_random_seed(2024)

        for epoch in range(self.args.n_epochs):
            self.run_epoch(epoch, A, X, S, R)


    def run_epoch(self, epoch, A, X, S, R):
        tf.random.set_random_seed(2024)

        loss, _ = self.session.run([self.loss, self.train_op],
                                         feed_dict={self.A: A,
                                                    self.X: X,
                                                    self.S: S,
                                                    self.R: R})
        print(loss)
        return loss

    def infer(self, A, X, S, R):
        tf.random.set_random_seed(2024)

        H, C = self.session.run([self.H, self.C],
                           feed_dict={self.A: A,
                                      self.X: X,
                                      self.S: S,
                                      self.R: R})

        return H, data_preprocess.conver_sparse_tf2np(C)




