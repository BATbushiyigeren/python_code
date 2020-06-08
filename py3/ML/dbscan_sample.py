# coding=utf-8
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


def achieve_data(dataset, path, label):
    if os.path.exists(path):
        filenames = os.listdir(path)
        for filename in filenames:
            data_np = np.genfromtxt(path + '/' + filename, dtype=float)
            if data_np.ndim == 1:
                data_np = np.expand_dims(data_np, axis=0)
            for data in data_np:
                dataset['value'].append(data)
                if label == 0:
                    dataset['lable'].append(0)
                else:
                    dataset['label'].append(1)
                dataset['file'].append(filename)


def dbscan(data):
    y_pred_color = []
    category = []
    fea_list = data['value']
    file_list = data['file']
    y_pred = DBSCAN(eps=0.5, min_samples=10).fit_predict(fea_list)
    for index, pred in enumerate(y_pred):
        if pred == -1:
            color = 'r'
        else:
            color = 'g'
        y_pred_color.append(color)


def main():
    white_path = sys.argv[1]
    black_path = sys.argv[2]
    dataset = dict({'value': [], 'label': [], 'file': []})
    dataset = achieve_data(dataset, white_path, 0)
    dataset = achieve_data(dataset, black_path, 1)


if __name__ == '__main__':
    main()
