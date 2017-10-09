import tkinter as tk
import numpy as np
import time
import matplotlib.pyplot as plt


class VarVisualizer:

    def __init__(self, name, data, size=(800, 600), framerate=24):


        self.size = size
        self.size_per_unit = [size[0]/len(data[0]), size[1]/len(data)]
        self.min_max = [np.min(data), np.max(data)]

        self.tk = tk.Tk(className=name)
        self.canvas = tk.Canvas(master=self.tk, width=size[0], height=size[1])
        self.canvas.pack()

        self.data = data
        self.name = name

        self.framerate = framerate/1000
        self.last_update = 0

        self._draw()

    def update_data(self, data):
        if (time.time() - self.last_update) > self.framerate:
            self.min_max = [np.min(data), np.max(data)]
            self.data = data
            self._draw()
            self.last_update = time.time()

    def _draw(self):
        self.canvas.delete('all')
        for y, r in enumerate(self.data):
            for x, w in enumerate(r):
                self._draw_unit(y, x, w)
        self.tk.update()

    def _draw_unit(self, y, x, w):
        x_magn = val_map(w, self.min_max[0], self.min_max[1], 0, self.size_per_unit[0])
        y_magn = val_map(w, self.min_max[0], self.min_max[1], 0, self.size_per_unit[1])

        x_pos = x*self.size_per_unit[0]+self.size_per_unit[0]/2-x_magn/2
        y_pos = y*self.size_per_unit[1]+self.size_per_unit[1]/2-y_magn/2

        self.canvas.create_rectangle(x_pos, y_pos, x_pos+x_magn, y_pos+y_magn, fill='red', outline=None)


def val_map(x, in_min, in_max,  out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


class ErrorVisualizer:

    def __init__(self, name, size=(800, 600), framerate=24):


        self.size = size

        self.fig = plt.figure()
        self.ln, = plt.plot([])
        plt.ion()
        plt.show()

        self.data = []
        self.name = name

        self.framerate = framerate/1000
        self.last_update = 0


    def update_data(self, data):
        self.data.append(1-data)
        if (time.time() - self.last_update) > self.framerate:
            self._draw()
            self.last_update = time.time()

    def _draw(self):
        plt.pause(.2)
        plt.gca().cla()
        plt.plot(range(len(self.data)), self.data)

        plt.draw()




    def _to_be_named(self):
        if len(self.data) <= self.size[0]:
            return self.data
        te = int(np.ceil(len(self.data)/self.size[0]))
        new = []
        for i in range(self.size[0]):
            new.append(sum(self.data[i*te:(i+1)*te])/(te+1))
        return new


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