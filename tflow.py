import tensorflow as tf
import numpy as np


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
                        output_activation, learning_rate, loss_function, model_path=''):

        self.session = None
        self.shape = shape
        self.model_path = model_path

        self.y_target = tf.placeholder(tf.float64, [None, shape[-1]])
        self.A = []
        self.x = tf.placeholder(tf.float64, [None, shape[0]])
        self.w = []
        self.b = []

        self._generate_weights(shape, init_value, init_dist)
        self._set_hidden_activations(hidden_activation)
        self._set_out_layer_activation(output_activation)

        self.error = AANN.LOSS_FN[loss_function](self.y_target, self.A[-1])
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='gradient_descent').minimize(self.error)

        correct_prediction = tf.equal(tf.argmax(self.A[-1], 1), tf.argmax(self.y_target, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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

    def batch_train(self, batch_x, batch_y):
        sess = self.get_session()
        return sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y_target: batch_y})

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


