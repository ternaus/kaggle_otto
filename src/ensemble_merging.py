from __future__ import division
import pandas as pd
import numpy as np

import xgboost
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from mlxtend.sklearn import EnsembleClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import theano
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

np.random.seed(123)

df = pd.DataFrame(columns=('w1', 'w2', 'w3', 'log_loss', 'std'))

#Best models are
# RandomForestClassifier
# GradientBoostingClassifier
# SVC
# Neural networks
# XGBoost

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

layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout', DropoutLayer),
           ('dense1', DenseLayer),
           ('output', DenseLayer)]

clf1 = RandomForestClassifier(n_estimators=500, n_jobs=-1)
clf2 = GradientBoostingClassifier(n_estimators=100)
clf3 = OneVsRestClassifier(SVC(C=5), n_jobs=-1)
clf4 = NeuralNet(layers=layers0,

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

params = {'max_depth': 10,
                    'n_estimators': 1000}
clf5 = xgboost.XGBClassifier(**params)


train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

target = train["target"].values
training = train.drop(["id", 'target'], 1).values
testing = test.drop("id", 1)

voting = 'soft'
# voting = 'hard'

# cv = 5
cv = 2

verbose = 2

i = 0
for w1 in range(1, 4):
    for w2 in range(1, 4):
        for w3 in range(1, 4):
            for w4 in range(1, 4):
                for w5 in range(1, 4):
                    # if len(set((w1, w2, w3, w4, w5))) == 1: # skip if all weights are equal
                    #     continue

                    eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3, clf4, clf5],
                                              voting=voting,
                                              weights=[w1, w2, w3, w4, w5])

                    scores = cross_validation.cross_val_score(
                                                    estimator=eclf,
                                                    X=X,
                                                    y=target,
                                                    cv=cv,
                                                    scoring='log_loss',
                                                    # n_jobs = -1,
                                                    verbose=verbose)

                    print (w1, w2, w3, w4, w5)
                    print 'score = ', scores.mean(), scores.std()

                    df.loc[i] = [w1, w2, w3, w4, w5, scores.mean(), scores.std()]
                    i += 1
            
df = df.sort(columns=['log_loss', 'std'], ascending=False)

print df

df.to_csv('logs/df_{n_folds}.csv'.format(n_folds=cv), index=False)