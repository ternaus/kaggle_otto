#!/usr/bin/env python
from __future__ import division
__author__ = 'Vladimir Iglovikov'

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import pandas as pd
from nolearn import lasagne
from sklearn.calibration import CalibratedClassifierCV
import os

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
import theano
import cPickle as pickle
import gzip

def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

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

X, target, encoder, scaler = load_train_data('../data/train.csv')
test, ids = load_test_data('../data/test.csv', scaler)


num_classes = len(encoder.classes_)
num_features = X.shape[1]


skf = cross_validation.StratifiedKFold(target, n_folds=10, random_state=42)

result = []

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
                 # update_learning_rate=0.001,
                 # update_momentum=0.9,
                update_momentum=theano.shared(float32(0.9)),
                 eval_size=0.001,
                 verbose=1,
                 max_epochs=100,
                 update_learning_rate=theano.shared(float32(0.03)),
                 on_epoch_finished=[
                    AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
                    AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ])


calibration_method = 'isotonic'
random_state = 42
n_folds = 10
method = 'nn_nfolds_{n_folds}_calibration_{calibration_method}'.format(n_folds=n_folds,
                                                                       calibration_method=calibration_method)

skf = cross_validation.StratifiedKFold(target,
                                       n_folds=n_folds,
                                       random_state=random_state)

ccv = CalibratedClassifierCV(base_estimator=clf,
                             method=calibration_method,
                             cv=skf)

fit = ccv.fit(X, target)

print 'predict on training set'
score = log_loss(target, fit.predict_proba(X))
print score

try:
    os.mkdir('logs')
except:
    pass

#save score to log
fName = open(os.path.join('logs', method + '.log'), 'w')
print >> fName, 'log_loss score on the training set is: ' + str(score)
fName.close()

print 'predict on testing'
prediction = ccv.predict_proba(test)
print 'saving prediction to file'
submission = pd.DataFrame(prediction)
submission.columns = ["Class_" + str(i) for i in range(1, 10)]
submission["id"] = test["id"]

try:
    os.mkdir('predictions')
except:
    pass

submission.to_csv(os.path.join('predictions', method + '.cvs'), index=False)

save_model = False

if save_model == True:
    print 'save model to file'
    try:
        os.mkdir('models')
    except:
        pass

    with gzip.GzipFile(os.path.join('models', method + '.pgz'), 'w') as f:
        pickle.dump(ccv, f)