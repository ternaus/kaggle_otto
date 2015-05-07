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

import gl_wrapper

def float32(k):
    return np.cast['float32'](k)

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()

class AdaptiveVariable(object):
    def __init__(self, name, start=0.03, stop=0.000001, inc=1.1, dec=0.5):
        self.name = name
        self.start, self.stop = start, stop
        self.inc, self.dec = inc, dec

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        if len(train_history) > 1:
            previous_valid = train_history[-2]['valid_loss']
        else:
            previous_valid = np.inf
        current_value = getattr(nn, self.name).get_value()
        if current_value < self.stop:
            raise StopIteration()
        if current_valid > previous_valid:
            getattr(nn, self.name).set_value(float32(current_value*self.dec))
        else:
            getattr(nn, self.name).set_value(float32(current_value*self.inc))

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
           ('dropout', DropoutLayer),
           ('dense2', DenseLayer),
           ('output', DenseLayer),
           ]

num_units = 1024

clf = NeuralNet(layers=layers0,

                 input_shape=(None, num_features),
                 dense0_num_units=num_units,
                 dropout_p=0.5,
                 dense1_num_units=num_units,
                 dense2_num_units=num_units,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,

                 update=nesterov_momentum,
                 # update_learning_rate=0.001,
                 # update_momentum=0.9,
                 update_momentum=theano.shared(float32(0.9)),
                 eval_size=0.1,
                 verbose=1,
                 max_epochs=1000,
                 update_learning_rate=theano.shared(float32(0.03)),
                 on_epoch_finished=[
                    AdaptiveVariable('update_learning_rate', start=0.001, stop=0.00001),
                    AdjustVariable('update_momentum', start=0.9, stop=0.999),
                    EarlyStopping(),
                ])


calibration_method = 'isotonic'
random_state = 42
n_folds = 10
method = 'nn_nfolds_{n_folds}_num_units{num_units}calibration_{calibration_method}'.format(n_folds=n_folds,
                                                                       calibration_method=calibration_method,
                                                                       num_units=num_units)

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
submission["id"] = ids

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