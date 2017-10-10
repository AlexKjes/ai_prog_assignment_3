from tflow import AANN
import numpy as np
from data_reader import DataSet



def auto_encoder():
    arr = np.array([[1 if i==j else 0 for j in range(10)] for i in range(10)])
    net = AANN([10, 3, 10], [0.1, 0.9], AANN.NORMAL,
               AANN.SIGMOID, AANN.SIGMOID,
               0.4, AANN.SQUARED_MEAN,
               visualize_free_variables=False, visualize_error=True)

    for _in in range(10000):
        net.batch_train(arr, arr)


    print(net.evaluate_network(arr, arr))
    print(net.feed_forward(arr))


def wine():
    data = DataSet('data_sets/winequality_red.txt', ';', [.8, .1, .1])
    net = AANN([data.features, 400, 200, 100, data.classes], [0.1, 0.9],
               AANN.NORMAL, AANN.SIGMOID, AANN.SOFTMAX, 0.001, AANN.CROSS_ENTROPY, model_path='models/wine2/')

    mini_batches = data.get_mini_batches(25)
    j = 1
    while True:
        for i in range(1000):
            for mb in mini_batches:
                net.batch_train(mb.x, mb.y)

        net.save_model()
        print("iteration " + str(j)+": " + str(net.evaluate_network(data.training.x, data.training.y)))
        j += 1
    #print(net.feed_forward(data.training.x[:10]))

def yeast():
    data = DataSet('data_sets/yeast.txt', ',', [.8, .1, .1])
    m = 6
    net = AANN([data.features, 2**m, 2**m, 2**m, 2**m,  data.classes], [0.1, 0.9],
               AANN.NORMAL, AANN.TANH, AANN.SOFTMAX, 0.05, AANN.CROSS_ENTROPY,
               model_path='', evaluate=True, visualize_error=True)

    mini_batches = data.get_mini_batches(150)
    j = 1
    while True:
        for i in range(100):
            for mb in mini_batches:
                net.batch_train(mb.x, mb.y, data.evaluation.x, data.evaluation.y)

        net.save_model()
        print("iteration " + str(j)+": " + str(net.evaluate_network(data.training.x, data.training.y)))
        j += 1
    #print(net.feed_forward(data.training.x[:10]))


yeast()
input()