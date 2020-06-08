# coding=utf-8
import os
import sys
import pickle
import sklearn
import numpy as np
from collections import Counter
from imblearn.combine import SMOTEENN
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, average_precision_score, f1_score, \
    confusion_matrix


class Config(object):
    seed = 7
    labels = ['white', 'CC']


def show_cm(cm, labels):
    percent = (cm*100.0)/np.array(np.matrix(cm.sum(axis=1)).T)
    print('Confusion Matrix Stats')
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            print("%s/%s: %.2f%% (%d/%d)" % (label_i, label_j, (percent[i][j]), cm[i][j], cm[i].sum()))


def save_model_to_disk(name, model, model_dir=''):
    serialized_model = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    model_path = os.path.join(model_dir, name+'.model')
    print('Storing Serialized Model to Disk (%s:%.2fMeg)' % (name, len(serialized_model)/1024.0/1024.0))
    open(model_path, 'wb').write(serialized_model)


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


def over_sample(x, y):
    smote_enn = SMOTEENN(sampling_strategy='auto', n_jobs=-1, random_state=Config.seed)
    x_resample, y_resample = smote_enn.fit_resample(x, y)
    print(sorted(Counter(y_resample).items()))
    return x_resample, y_resample


def train_test_model(x, y):
    x, y = over_sample(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, oob_score=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    scores = cross_val_score(clf, x, y, cv=skf, scoring='accuracy')
    print(scores)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    report_evaluation_metrics(y_test, y_pred)
    # 返回一个混淆矩阵
    # cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    # show_cm(cm, Config.labels)
    return clf


def report_evaluation_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred, labels=[0, 1], pos_label=1)
    recall = recall_score(y_true=y_true, y_pred=y_pred, labels=[0, 1], pos_label=1)
    average_precision = average_precision_score(y_true=y_true, y_score=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=[0, 1], pos_label=1)
    conf_matrix = confusion_matrix(y_true, y_pred)
    print('Accuracy: {0:0.2f}'.format(accuracy))
    print('Precision: {0:0.2f}'.format(precision))
    print('Recall: {0:0.2f}'.format(recall))
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    print('F1: {0:0.2f}'.format(f1))
    print("confusion matrix:", conf_matrix)


def grid_search_cv(x, y):
    param_grid = {'n_estimators': range(80, 120, 10), 'max_depth': range(10, 80, 10)}
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, oob_score=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=skf, scoring='roc_auc')
    grid.fit(x, y)
    print(grid.grid_scores_)
    print('best score: %.3f' % grid.best_score_)
    print('best params: %s' % grid.best_params_)


def main():
    if len(sys.argv) < 3:
        print('Missing Parameters')
        print('1 positive np file; 2 negative np file;3 model name')
        exit(0)
    white_file = sys.argv[1]
    black_file = sys.argv[2]
    model_name = sys.argv[3]
    data_x = []
    data_y = []
    data_x, data_y = achieve_data(white_file, 0, data_x, data_y)
    data_x, data_y = achieve_data(black_file, 1, data_x, data_y)
    model = train_test_model(data_x, data_y)
    save_model_to_disk(model_name, model)


if __name__ == '__main__':
    main()

