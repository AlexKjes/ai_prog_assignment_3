import tensorflow as tf
import numpy as np
import nn_visualizer as visual
import re
import problem_generators as pg#13
from data_reader import DataSet as Ds
import time
import tflowtools as tft

class ANN:

    NORMAL = 1
    EQUIPROBABLE = 2

    SIGMOID = 0
    TANH = 1
    RELU = 2
    ELU = 3
    SOFTMAX = 4
    ANALYTIC = 5

    SQUARED_MEAN = 0
    CROSS_ENTROPY = 1

    ACTIVATION_FN = [
        tf.nn.sigmoid,
        tf.nn.tanh,
        tf.nn.relu6,
        tf.nn.elu,
        tf.nn.softmax,
        lambda x: tf.log(1 + tf.exp(x))
        ]

    LOSS_FN = [
        lambda y_target, y: tf.reduce_mean(0.5*(y_target - y)**2, reduction_indices=1),
        lambda y_target, y:tf.reduce_mean(-tf.reduce_sum(y_target * tf.log(y), reduction_indices=[1])),
    ]

    def __init__(self, shape, init_value, init_dist, hidden_activation,
                 output_activation, learning_rate, loss_function, model_path=''):

        self.session = None  # active tf session
        self.shape = shape  # shape of network
        self.model_path = model_path  # save/load path for model

        # Model variables and operations
        self.y_target = tf.placeholder(tf.float64, [None, shape[-1]])  # target output
        self.A = []  # activations
        self.x = tf.placeholder(tf.float64, [None, shape[0]])  # net input
        self.w = []  # all weights
        self.b = []  # all biases

        # initialize network
        self._generate_weights(shape, init_value, init_dist)
        self._set_hidden_activations(hidden_activation)
        self._set_out_layer_activation(output_activation)

        self.error = ANN.LOSS_FN[loss_function](self.y_target, self.A[-1])
        self.learning_rate = learning_rate
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='gradient_descent').minimize(self.error)

        # Testing network accuracy currently only for classification. TODO add support for regression
        correct_prediction = tf.equal(tf.argmax(self.A[-1], 1), tf.argmax(self.y_target, 1))
        self.accuracy = 1-tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # create tf saver. used for network loading and saving
        self.saver = tf.train.Saver()

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='gradient_descent').minimize(self.error)

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

    def get_vars(self, var_list):
        return self.get_session().run(var_list)

    def save_model(self, layer_range=None):
        if layer_range is None:
            layer_range = [0, len(self.w)]
        layers = {}
        for i in range(layer_range[0], layer_range[1]):
            layers['w'+str(i)] = self.w[i]
            layers['b'+str(i)] = self.b[i]
        if len(self.model_path) != 0:
            self.saver.save(self.get_session(), self.model_path)

    def batch_train(self, batch_x, batch_y, return_variables = []):
        sess = self.get_session()
        ret = sess.run([self.optimizer] + return_variables,
                                 feed_dict={self.x: batch_x, self.y_target: batch_y})
        return ret[1:] if isinstance(ret, list) and len(ret)>1 else None

    def feed_forward(self, x):
        sess = self.get_session()
        return sess.run(self.A[-1], feed_dict={self.x: x})

    def custom_run(self, operations, feed_dict):
        return self.get_session().run(operations, feed_dict=feed_dict)

    def evaluate_network(self, batch_x, batch_y):
        sess = self.get_session()
        return sess.run(self.accuracy, feed_dict={self.x: batch_x, self.y_target: batch_y})

    def _generate_weights(self, shape, init_value, init_dist):
        for i in range(1, len(shape)):
            w = None
            b = None
            if init_dist == ANN.NORMAL:
                w = np.random.randn(shape[i-1], shape[i])/(init_value[0] - init_value[1]) - init_value[0]
                b = np.random.randn(shape[i])/(init_value[0] - init_value[1]) - init_value[0]

            if init_dist == ANN.EQUIPROBABLE:
                w = np.random.uniform(init_value[0], init_value[1], (shape[i-1], shape[i]))
                b = np.random.random_integers(init_value[0], init_value[1], (shape[i], 1))
            self.w.append(tf.Variable(w, name='w' + str(i)))
            self.b.append(tf.Variable(b, name='b' + str(i)))

    def _set_hidden_activations(self, activation):
        z = tf.matmul(self.x, self.w[0]) + self.b[0]
        a = ANN.ACTIVATION_FN[activation](z)
        self.A.append(a)
        for i in range(1, len(self.w)-1):
            z = tf.matmul(a, self.w[i]) + self.b[i]
            a = ANN.ACTIVATION_FN[activation](z)
            self.A.append(a)

    def _set_out_layer_activation(self, activation):
        z = tf.matmul(self.A[-1], self.w[-1]) + self.b[-1]
        a = ANN.ACTIVATION_FN[activation](z)
        self.A.append(a)


class NeuralMan:

    default = {
        'data': 'data_sets/glass.txt',
        'data_delimiter': ',',
        'hidden_layers': [100],
        'loss_fn': 1,
        'output_fn': 3,
        'hidden_fn': 0,
        'learning_rate': 0.1,
        'weight_init_range': [.1, .9],
        'model_path': '',
        'evaluate': True,
        'visualize_error': True,
        'steps': 0,
        'shuffle_data': False,
        'evaluation_step': 100,
        'error_limit': 1,
        'map_batch_size': 0,
        'normalize_data': True
    }

    def __init__(self, conf_file_path):
        self.properties = NeuralMan.default
        self.read_conf_file(conf_file_path)
        self.data_set = self.data_resolver()
        self.net = self.make_nn()

        self.mb_error = []
        self.training_error = []
        self.evaluation_error = []
        self.test_error = 0

        self.weight_visualizers = None
        self.bias_visualizers = None
        self.error_visualizer = visual.ErrorVisualizer('Loss') if self.properties['visualize_error'] else None

        self.train()

    def train(self):
        mini_batches = self.data_set.get_mini_batches(self.properties['mini_batch_size'])
        epoch = 0
        err = 1
        start_time = time.time()
        while epoch < self.properties['steps'] or self.properties['steps'] == 0 and err >= self.properties['error_limit']:
            for mb in mini_batches:
                self._train_return_handler(self.net.batch_train(mb.x, mb.y, self._net_get()))

            # End of epoch stuff
            self.training_error.append(sum(self.mb_error)/len(self.mb_error))
            self.mb_error = []
            err = self.training_error[-1]
            epoch += 1
            if self.properties['visualize_error']:
                self.error_visualizer.update_training_error(self.training_error,
                                                            np.arange(0, len(self.training_error), 1))
            # Validation
            if epoch % self.properties['evaluation_step'] == 0 and self.properties['evaluate']:
                self.evaluation_error.append(self.net.evaluate_network(self.data_set.evaluation.x,
                                                                       self.data_set.evaluation.y))
                if self.properties['visualize_error']:
                    self.error_visualizer.update_evaluation_error(self.evaluation_error,
                                                                  np.arange(0, epoch, self.properties['evaluation_step']))
                print(str(round(time.time()-start_time, 2)) + "s : " + str(self.training_error[-1]))

        self._after_training(epoch, time.time()-start_time)

    def read_conf_file(self, path):
        with open(path, 'r') as f:
            for line in f:
                if line != '\n' or line[0] != '#':
                    kv = re.split(':| ', line[:-1])
                    key = kv[0].strip()
                    value = []
                    [value.append(v)if len(v)else None for v in kv[1:]]
                    if key == 'data':
                        if value[0].isnumeric():
                            self.properties[key] = int(value[0])
                        else:
                            self.properties[key] = value[0]
                    elif key == 'data_distribution' or \
                            key == 'weight_init_range':
                        self.properties[key] = [float(x) for x in value]
                    elif key == 'hidden_layers' or \
                        key == 'map_layers' or \
                        key == 'map_dendrograms' or \
                        key == 'display_weights_in_training' or \
                        key == 'display_biases_in_training' or \
                        key == 'display_weights_after_training' or \
                            key == 'display_biases_after_training':
                        self.properties[key] = [int(v) for v in value]
                    elif key == 'loss_fn' or \
                        key == 'output_fn' or \
                        key == 'hidden_fn' or \
                        key == 'map_batch_size' or \
                        key == 'mini_batch_size' or \
                        key == 'steps' or \
                        key == 'shuffle_data' or \
                        key == 'evaluation_step' or \
                            key == 'save':
                        self.properties[key] = int(value[0])
                    elif key == 'learning_rate' or \
                            key == 'error_limit':
                        self.properties[key] = float(value[0])
                    elif key == 'model_path' or \
                            key == 'data_delimiter':
                        self.properties[key] = value[0]
                    elif key == 'shuffle_data' or \
                        key == 'visualize_error' or \
                            key == 'normalize_data':
                        self.properties[key] = int(value[0]) > 0

    def data_resolver(self):
        dd = self.properties['data_distribution']
        if isinstance(self.properties['data'], int):
            return pg(int(self.properties['data']), dd)
        else:
            return Ds(self.properties['data'], self.properties['data_delimiter'], dd,
                      self.properties['shuffle_data'], normalize=self.properties['normalize_data'])

    def make_nn(self):
        return ANN([self.data_set.features] + self.properties['hidden_layers'] + [self.data_set.classes],
                   [float(x) for x in self.properties['weight_init_range']],
                   ANN.NORMAL,
                   int(self.properties['hidden_fn']),
                   int(self.properties['output_fn']),
                   float(self.properties['learning_rate']),
                   int(self.properties['loss_fn']),
                   self.properties['model_path'])

    def _after_training(self, n_mini_batches, train_time):
        self.test_err = self.net.evaluate_network(self.data_set.test.x, self.data_set.test.y)
        train_err = self.net.evaluate_network(self.data_set.training.x, self.data_set.training.y)
        print('\n\n\n\n\n')

        print('Training took {} seconds'.format(round(train_time, 2)))
        print('Trained on {} epochs'.format(n_mini_batches))
        print('Correct classifications on test set: {}%'.format(round(self.test_err*100, 2)))
        print('Correct classifications on training set: {}%'.format(round(train_err*100, 2)))
        print('Correct classifications on evaluation set: {}%'.format(round(self.evaluation_error[-1]*100, 2)))

        self._visualize_after_run()



        input('Press enter to end')

    def _train_return_handler(self, train_return):
        i = 1
        self.mb_error.append(train_return[0])
        if 'display_weights_in_training' in self.properties.keys():
            dw = self.properties['display_weights_in_training']
            weights = train_return[i:i + len(dw)]
            i += len(dw)
            if self.weight_visualizers is None:
                self.weight_visualizers = [visual.VarVisualizer('W' + str(x), w) for x, w in zip(dw, weights)]
            else:
                [vv.update_data(w) for vv, w in zip(self.weight_visualizers, weights)]

        if 'display_biases_in_training' in self.properties.keys():
            db = self.properties['display_biases_in_training']
            biases = [b.reshape(b.shape[0], 1).T for b in train_return[i:i + len(db)]]
            if self.bias_visualizers is None:
                self.bias_visualizers = [visual.VarVisualizer('B' + str(x), b) for x, b in zip(db, biases)]
            else:
                [bv.update_data(b) for bv, b in zip(self.bias_visualizers, biases)]



    def _net_get(self):
        ret = [self.net.accuracy]
        if 'display_weights' in self.properties.keys():
            [ret.append(self.net.w[x-1]) for x in self.properties['display_weights_in_training']]
        if 'display_biases' in self.properties.keys():
            [ret.append(self.net.b[x-1]) for x in self.properties['display_biases_in_training']]

        return ret

    def _visualize_after_run(self):
        self._visualize_after_vars()
        self._visualize_after_error()
        self._visualize_after_activations()
        self._visualize_dendrogram()

    def _visualize_after_activations(self):
        if self.properties['map_batch_size'] > 0:
            if 'map_layers' in self.properties.keys():
                act = self.net.custom_run([self.net.A[a-1] for a in self.properties['map_layers']],
                                          {self.net.x: self.data_set.training.x[0:self.properties['map_batch_size']]})
                for am, al in zip(act, self.properties['map_layers']):
                    visual.VarVisualizer('A'+str(al), am)


    def _visualize_after_error(self):
        if not self.properties['visualize_error']:
            self.error_visualizer = visual.ErrorVisualizer('Error')
            x = np.arange(0, len(self.training_error), self.properties['evaluation_step'])
            self.error_visualizer.update_evaluation_error(self.evaluation_error, x[:len(self.evaluation_error)])
            self.error_visualizer.update_training_error(self.training_error[::self.properties['evaluation_step']], x)
        self.error_visualizer.plot_test([self.test_err, self.test_err], [0, len(self.training_error)])

    def _visualize_after_vars(self):
        if 'display_weights_after_training' in self.properties.keys():
            w_layers = self.properties['display_weights_after_training']
            weights = self.net.get_vars([self.net.w[w-1] for w in w_layers])
            for wm, wl in zip(weights, w_layers):
                visual.VarVisualizer('W' + str(wl), wm)

        if 'display_biases_after_training' in self.properties.keys():
            b_layers = self.properties['display_biases_after_training']
            biases = self.net.get_vars([self.net.b[b-1] for b in b_layers])
            for bm, bl in zip(biases, b_layers):
                visual.VarVisualizer('B' + str(bl), bm.reshape(bm.shape[0], 1).T)

    def _visualize_dendrogram(self):
        if 'map_dendrograms' in self.properties.keys() and self.properties['map_batch_size'] > 0:
            act = self.net.custom_run([self.net.A[a - 1] for a in self.properties['map_dendrograms']],
                                      {self.net.x: self.data_set.training.x[0:self.properties['map_batch_size']]})
            for am, al in zip(act, self.data_set.training.x[0:self.properties['map_batch_size']]):
                print('YOLO!!!!!!')
                print(al)
                tft.dendrogram(am, tft.bits_to_str(al))
