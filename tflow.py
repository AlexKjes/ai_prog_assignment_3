import tensorflow as tf
import numpy as np
import nn_visualizer as visual

class AANN:

    NORMAL = 1
    EQUIPROBABLE = 2

    SIGMOID = 0
    TANH = 1
    RELU = 2
    SOFTMAX = 3
    ANALYTIC = 4

    SQUARED_MEAN = 0
    CROSS_ENTROPY = 1

    ACTIVATION_FN = [
        tf.nn.sigmoid,
        tf.nn.tanh,
        tf.nn.relu6,
        tf.nn.softmax,
        lambda x: tf.log(1 + tf.exp(x))
        ]

    LOSS_FN = [
        lambda y_target, y: tf.reduce_mean(0.5*(y_target - y)**2, reduction_indices=1),
        lambda y_target, y:tf.reduce_mean(-tf.reduce_sum(y_target * tf.log(y), reduction_indices=[1])),
    ]

    def __init__(self, shape, init_value, init_dist, hidden_activation,
                output_activation, learning_rate, loss_function, model_path='',
                evaluate=False, visualize_free_variables=False, visualize_error=False):

        self.session = None  # active tf session
        self.shape = shape  # shape of network
        self.model_path = model_path  # save/load path for model

        # Model variables and operations
        self.y_target = tf.placeholder(tf.float64, [None, shape[-1]])  # target output
        self.A = []  # activations
        self.x = tf.placeholder(tf.float64, [None, shape[0]])  # net input
        self.w = []  # all weights
        self.b = []  # all biases

        # Visualizers
        self.visualize_vars = visualize_free_variables
        self.visualize_error = visualize_error
        self.wnb_visualizers = []  # visualizers for weights and biases
        self.error_visualizer = visual.ErrorVisualizer('Error', ee=evaluate) if visualize_error else None

        # initialize network
        self._generate_weights(shape, init_value, init_dist)
        self._set_hidden_activations(hidden_activation)
        self._set_out_layer_activation(output_activation)

        self.error = AANN.LOSS_FN[loss_function](self.y_target, self.A[-1])
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='gradient_descent').minimize(self.error)

        # Testing network accuracy currently only for classification. TODO add support for regression
        correct_prediction = tf.equal(tf.argmax(self.A[-1], 1), tf.argmax(self.y_target, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # create tf saver. used for network loading and saving
        self.saver = tf.train.Saver()



    def get_session(self):
        if self.session is None:
            self.session = tf.InteractiveSession()
            if len(self.model_path) != 0:
                try:
                    self.saver.restore(self.session, self.model_path)
                except:
                    tf.global_variables_initializer().run()
            else:
                tf.global_variables_initializer().run()
        return self.session

    def save_model(self, layer_range=None):
        if layer_range is None:
            layer_range = [0, len(self.w)]
        layers = {}
        for i in range(layer_range[0], layer_range[1]):
            layers['w'+str(i)] = self.w[i]
            layers['b'+str(i)] = self.b[i]
        if len(self.model_path) != 0:
            self.saver.save(self.get_session(), self.model_path)

    def batch_train(self, batch_x, batch_y, train_x=None, train_y=None):
        sess = self.get_session()
        _, ws, bs, te = sess.run([self.optimizer, self.w, self.b, self.accuracy], feed_dict={self.x: batch_x, self.y_target: batch_y})
        ee = None
        if train_x is not None:
            ee = sess.run(self.accuracy, feed_dict={self.x: train_x, self.y_target: train_y})
        self._visualize(ws, bs, te, ee)


    def feed_forward(self, input):
        sess = self.get_session()
        return sess.run(self.A[-1], feed_dict={self.x: input})

    def evaluate_network(self, batch_x, batch_y):
        sess = self.get_session()
        return sess.run(self.accuracy, feed_dict={self.x: batch_x, self.y_target: batch_y})

    def _generate_weights(self, shape, init_value, init_dist):
        for i in range(1, len(shape)):
            w = None
            b = None
            if init_dist == AANN.NORMAL:
                w = np.random.randn(shape[i-1], shape[i])/(init_value[0] - init_value[1]) - init_value[0]
                b = np.random.randn(shape[i])/(init_value[0] - init_value[1]) - init_value[0]

            if init_dist == AANN.EQUIPROBABLE:
                w = np.random.uniform(init_value[0], init_value[1], (shape[i-1], shape[i]))
                b = np.random.random_integers(init_value[0], init_value[1], (shape[i], 1))
            self.w.append(tf.Variable(w, name='w' + str(i)))
            self.b.append(tf.Variable(b, name='b' + str(i)))
            # Set var visualizers
            if self.visualize_vars:
                self.wnb_visualizers.append(visual.VarVisualizer('layer' + str(i),
                                            np.concatenate((w.T, b.reshape(b.shape[0], 1)), axis=1)))

    def _set_hidden_activations(self, activation):
        z = tf.matmul(self.x, self.w[0]) + self.b[0]
        a = AANN.ACTIVATION_FN[activation](z)
        self.A.append(a)
        for i in range(1, len(self.w)-1):
            z = tf.matmul(a, self.w[i]) + self.b[i]
            a = AANN.ACTIVATION_FN[activation](z)
            self.A.append(a)

    def _set_out_layer_activation(self, activation):
        z = tf.matmul(self.A[-1], self.w[-1]) + self.b[-1]
        a = AANN.ACTIVATION_FN[activation](z)
        self.A.append(a)

    def _visualize(self, ws, bs, e, te=None):
        if self.visualize_vars:
            [v.update_data(np.concatenate((w.T, b.reshape(b.shape[0], 1)), axis=1)) for v, w, b in
             zip(self.wnb_visualizers, ws, bs)]
        if self.visualize_error:
            self.error_visualizer.update_error(e, te)

