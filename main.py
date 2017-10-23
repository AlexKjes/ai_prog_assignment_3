from tflow import *
import numpy as np
from data_reader import DataSet
import tflowtools as tft



"""
def auto_encoder():
    arr = np.array([[1 if i==j else 0 for j in range(10)] for i in range(10)])
    net = ANN([10, 3, 10], [0.1, 0.9], ANN.NORMAL,
              ANN.SIGMOID, ANN.SIGMOID,
              0.4, ANN.SQUARED_MEAN,
              visualize_free_variables=False, visualize_error=True)

    for _in in range(10000):
        net.batch_train(arr, arr)


    print(net.evaluate_network(arr, arr))
    print(net.feed_forward(arr))


def wine():
    data = DataSet('data_sets/winequality_red.txt', ';', [.8, .1, .1])
    net = ANN([data.features, 512, 512, 512, data.classes], [0.1, 0.9],
              ANN.NORMAL, ANN.SIGMOID, ANN.SOFTMAX, 0.02, ANN.CROSS_ENTROPY,
              model_path='', visualize_error=True, evaluate=True)

    mini_batches = data.get_mini_batches(150)
    j = 1
    while True:
        for i in range(100):
            for mb in mini_batches:
                net.batch_train(mb.x, mb.y, data.evaluation.x, data.evaluation.y)

        net.save_model()
        print("iteration " + str(j)+": " + str(net.evaluate_network(data.training.x, data.training.y)))
        j += 1


def yeast():
    data = DataSet('data_sets/yeast.txt', ',', [.8, .1, .1])
    m = 9
    net = ANN([data.features, 2 ** m, 2 ** m, 2 ** m, data.classes], [0.1, 0.9],
              ANN.NORMAL, ANN.TANH, ANN.SOFTMAX, 0.006, ANN.CROSS_ENTROPY,
              model_path='', evaluate=True, visualize_error=False)

    mini_batches = data.get_mini_batches(150)
    j = 1
    while True:
        for i in range(100):
            for mb in mini_batches:
                net.batch_train(mb.x, mb.y, data.evaluation.x, data.evaluation.y)

        net.save_model()
        print("iteration " + str(j)+": " + str(net.evaluate_network(data.training.x, data.training.y)))
        j += 1


def glass():
    data = DataSet('data_sets/glass.txt', ',', [.8, .1, .1], random=False)


    f = 4
    data.training.x = pca(data.training.x, f)
    data.evaluation.x = pca(data.evaluation.x, f)

    mini_batches = data.get_mini_batches(10)



    net = ANN([f, 200, data.classes], [0.1, 0.2],
              ANN.NORMAL, ANN.SIGMOID, ANN.SOFTMAX, 0.001, ANN.CROSS_ENTROPY,
              model_path='', visualize_error=False, evaluate=True)




    j = 1
    while True:
        for i in range(100):
            for mb in mini_batches:
                net.batch_train(mb.x, mb.y, data.evaluation.x, data.evaluation.y)


        #net.save_model()
        print("iteration " + str(j)+": " + str(net.evaluate_network(data.training.x, data.training.y)))
        j += 1



glass()

input()
"""


NeuralMan('configs/bit_counter.txt')



"""
sets = tft.gen_all_parity_cases(10, False)
with open('data_sets/parity.txt', 'w') as f:


sets = tft.gen_segmented_vector_cases(25,1000,0,8, poptargs=False)
with open('data_sets/segment_counter.txt', 'w') as f:

    for s in sets:
        l = ''
        for b in s[0]:
            l += str(b) + ','
        l += str(s[1]) + '\n'
        f.write(l)

"""