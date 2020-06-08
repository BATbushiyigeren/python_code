# coding=utf-8
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def achieve_data(path, label, x, y):
    if os.path.exists(path):
        filenames = os.listdir(path)
        for filename in filenames:
            if filename.endswith(r'.txt'):
                data_np = np.genfromtxt(path+'/'+filename)
                if data_np.ndim == 1:
                    data_np = np.expand_dims(data_np, axis=0)
                x.extend(data_np)
                y.extend(label for i in range(len(data_np)))
    return x, y


def tsne(fea, label):
    tsne = TSNE(n_components=2)
    fea_tsne = tsne.fit_transform(fea)
    x = fea_tsne[:, 0]
    y = fea_tsne[:, 1]
    plt.scatter(x, y, c=label)
    plt.show()


def main():
    white_file = sys.argv[1]
    black_file = sys.argv[2]
    data_x = []
    data_y = []
    data_x, data_y = achieve_data(white_file, 0, data_x, data_y)
    data_x, data_y = achieve_data(black_file, 1, data_x, data_y)
    tsne(data_x, data_y)


if __name__ == '__main__':
    main()
