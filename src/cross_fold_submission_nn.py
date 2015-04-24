#!/usr/bin/env python
from __future__ import division
__author__ = 'Vladimir Iglovikov'

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import pandas as pd
from nolearn import lasagne

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

def load_train_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, encoder, scaler

def load_test_data(path, scaler):
    df = pd.read_csv(path)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    X = scaler.transform(X)
    return X, ids

X, y, encoder, scaler = load_train_data('../data/train.csv')
test, ids = load_test_data('../data/test.csv', scaler)


num_classes = len(encoder.classes_)
num_features = X.shape[1]


skf = cross_validation.StratifiedKFold(y, n_folds=10, random_state=42)

result = []
ind = 1
for train_index, test_index in skf:
    layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout', DropoutLayer),
           ('dense1', DenseLayer),
           ('output', DenseLayer)]
    clf = NeuralNet(layers=layers0,

                 input_shape=(None, num_features),
                 dense0_num_units=512,
                 dropout_p=0.5,
                 dense1_num_units=512,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,

                 update=nesterov_momentum,
                 update_learning_rate=0.001,
                 update_momentum=0.9,

                 eval_size=None,
                 verbose=1,
                 max_epochs=300)


    X_train, X_test = X[train_index], y[train_index]

    fit = clf.fit(X_train, X_test)

    prediction_1 = fit.predict_proba(X[test_index])
    a = log_loss(y[test_index], prediction_1)
    print a
    result += [a]

    prediction_2 = fit.predict_proba(test)
    submission = pd.DataFrame(prediction_2)
    submission.columns = ["Class_" + str(i) for i in range(1, 10)]
    submission["id"] = ids
    submission.to_csv("nn_mi_300_512_ulr0001_cv10_ind{ind}.csv".format(ind=ind), index=False)
    ind += 1

print np.mean(a)