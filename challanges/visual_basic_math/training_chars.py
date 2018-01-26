import imageio
import numpy as np
import os
import random

chars = [
    '0', '1', '2', '3', '4',
    '5', '6', '7', '8', '9',
    'div', 'mul', 'plus', 'minus'
]


def read_image(i):
    i = imageio.imread(i)
    i = i.flatten().astype(np.float64)

    for p in range(0, i.shape[0]):
      i[p] = (255 - i[p]) / 255

    return i


def read_dir(d):
    r = []

    for f in filter(lambda x: x.endswith('.png'), os.listdir(d)):
        r.append(read_image(os.path.join(d, f)))

    return r


def pairwith(x, y_):
    return [(i, y_) for i in x]


def load_chars(d):
    r = {}

    for i, c in enumerate(chars):
        y_ = np.zeros(len(chars))
        y_[i] = 1

        r[c] = pairwith(read_dir(os.path.join(d,c)), y_)

    return r
        


class Chars:
    def __init__(self, d):
        train = []
        test = []

        all_chars = load_chars(d)

        for chars in all_chars.values():
            l = len(chars)

            train += chars[:int(l*0.8)]
            test += chars[int(l*0.8):]

        random.shuffle(train)
        random.shuffle(test)

        self.train = Dataset(np.array(train))
        self.test = Dataset(np.array(test))


class Dataset:
    def __init__(self, data):
        self.data = data
        self.index = 0

    def next_batch(self, n):
        batch = self.data[self.index:self.index+n]
        self.index += n

        if self.index >= len(self.data):
            self.index = 0

        x = np.array([b[0] for b in batch])
        y = np.array([b[1] for b in batch])

        return x, y

    @property
    def images(self):
        return np.array([b[0] for b in self.data])

    @property
    def labels(self):
        return np.array([b[1] for b in self.data])

if __name__ == '__main__':
    c = Chars('./chars/')

    b = c.train.next_batch(10)

    print(b[0])
    print(b[1])
