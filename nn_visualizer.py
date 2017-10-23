import tkinter as tk
import numpy as np
import time
import matplotlib.pyplot as plt



class VarVisualizer:

    def __init__(self, name, data, size=(800, 800)):
        data = data.T

        sr = [data.shape[1]/data.shape[0], data.shape[0]/data.shape[1]]
        self.size = [size[1]*(sr[0] if sr[0] < 1 else 1),
                     size[0]*(sr[1] if sr[1] < 1 else 1)]

        self.size_per_unit = [self.size[0]/len(data[0]), self.size[1]/len(data)]
        self.min_max = [abs(np.min(data)), np.max(data)]

        self.tk = tk.Tk(className=name)
        self.canvas = tk.Canvas(master=self.tk, width=self.size[0], height=self.size[1])
        self.canvas.pack()
        self.data = data
        self.name = name

        self._draw()

    def update_data(self, data):
        data = data.T
        self.min_max = [np.min(data), np.max(data)]
        self.data = data
        self._draw()

    def _draw(self):
        self.canvas.delete('all')
        for y, r in enumerate(self.data):
            for x, w in enumerate(r):
                self._draw_unit(y, x, w)

        self.tk.update()

    def _draw_unit(self, y, x, w):
        x_magn = val_map(w, 0, self.min_max[1] if w > 0 else self.min_max[0], 0, self.size_per_unit[0])
        y_magn = val_map(w, 0, self.min_max[1] if w > 0 else self.min_max[0], 0, self.size_per_unit[1])

        x_pos = x*self.size_per_unit[0]+self.size_per_unit[0]/2-x_magn/2
        y_pos = y*self.size_per_unit[1]+self.size_per_unit[1]/2-y_magn/2

        self.canvas.create_rectangle(x_pos, y_pos, x_pos+x_magn, y_pos+y_magn,
                                     fill=('red' if w > 0 else 'blue'), outline=None)


def val_map(x, in_min, in_max,  out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


class ErrorVisualizer:

    def __init__(self, name):

        self.fig = plt.figure(name)
        plt.ylabel('%error')
        plt.ylim((0, 1))
        self.training_error = plt.plot([], [], 'b-', label='Training Error')[0]
        self.evaluation_error = plt.plot([], [], 'r--', label='Evaluation Error')[0]
        self.test_error = plt.plot([], [], 'g-', label='Test Error')[0]
        plt.legend()
        plt.ion()
        plt.show()

        self.counter = 0


    def update_training_error(self, y, x):
        self.training_error.set_xdata(x)
        self.training_error.set_ydata(y)
        plt.xlim(0, x[-1])
        plt.pause(0.005)

    def update_evaluation_error(self, y, x):
        self.evaluation_error.set_xdata(x)
        self.evaluation_error.set_ydata(y)
        plt.pause(0.005)

    def plot_test(self, y, x):
        self.test_error.set_xdata(x)
        self.test_error.set_ydata(y)
        plt.pause(0.005)

if __name__ == '__main__':


    """
    data = np.arange(0, 25, 1, dtype=np.int64).reshape((5, 5))
    direction = np.ones((5, 5), dtype=np.int64)
    direction[0][0] = -1
    visual = VarVisualizer('test', data, [800, 600])

    while True:
        start = time.time()
        limit = np.equal(data, 24) + np.equal(data, 0)
        direction = np.where(limit, direction*-1, direction)
        data += direction
        visual.update_data(data)
        sleep_time = (60 - (time.time() - start))/1000
        time.sleep(sleep_time)
    """

    vis = ErrorVisualizer('yolo')
    for i in range(100):
        vis.update_data(i)
        time.sleep(0.2)