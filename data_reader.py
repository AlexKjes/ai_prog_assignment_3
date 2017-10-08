import numpy as np


class DataSet:

    def __init__(self, file_name, delimiter, set_dist):
        self.features = 0
        self.classes = 0

        self.training = SubSet([], [])
        self.evaluation = SubSet([], [])
        self.test = SubSet([], [])

        sets = self.__read_file(file_name, delimiter)
        self.__distribute_sets(sets, set_dist)

    def get_mini_batches(self, batch_size):
        ret = []
        n_its = len(self.training)//batch_size
        for i in range(n_its):
            if i < n_its-1:
                ret.append(SubSet(self.training.x[i*batch_size:(i+1)*batch_size],
                                  self.training.y[i*batch_size:(i+1)*batch_size]))
            else:
                ret.append(SubSet(self.training.x[i*batch_size:], self.training.y[i*batch_size:]))
        return ret

    def __read_file(self, file_name, delimiter):
        data = []
        size = 0
        out_max = 0
        with open(file_name, 'r') as f:
            for i, line in enumerate(f):
                sline = line.split(delimiter)
                data.append([float(l)for l in sline[:-1]] + [int(sline[-1])])
                size += 1
                if data[-1][-1] > out_max:
                    out_max = data[-1][-1]
        self.classes = out_max+1
        self.features = len(data[0])-1
        return data

    def __distribute_sets(self, data, set_dist):
        sets = [self.training, self.evaluation, self.test]
        for d in data:
            s = np.random.choice(np.arange(0, 3, dtype=np.int8), p=set_dist)

            sets[s].x.append(d[:-1])
            sets[s].y.append(self.int_to_one_hot(d[-1], self.classes))

    @staticmethod
    def int_to_one_hot(value, size):
        ret = np.zeros(size)
        ret[value] = 1
        return ret


class SubSet:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)


