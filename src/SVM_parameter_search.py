#!/usr/bin/env python
__author__ = 'Vladimir Iglovikov'

'''
This script will do randomized search to find the best or almost the best parameters for this problem
for sklearn package
'''
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

import xgboost as xgb
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn import cross_validation
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
import math
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from operator import itemgetter
# Utility function to report best scores
def report(grid_scores, n_top=10):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

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

from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# clf = XGBClassifier()
from scipy.stats import randint as sp_randint

# print help(clf)
# param_dist = {"n_estimators": [100, 200],
#               "max_depth": [None, 8, 10, 12, 20],
#               'learning_rate': [0.1],
#               # "max_features": sp_randint(1, 11),
#               # "min_samples_split": sp_randint(1, 11),
#               # "min_samples_leaf": sp_randint(1, 11),
#               }


random_state = 42
n_folds = 2

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_dist = dict(gamma=gamma_range, C=C_range)

clf = SVC(probability=True, cache_size=2048)
# searhc = GridSearchCV(clf, param_dist, random_state=42, cv=2, scoring='log_loss', verbose=3, n_jobs=-1)
skf = cross_validation.StratifiedKFold(target,
                                       n_folds=n_folds,
                                       random_state=random_state)
# cv = StratifiedShuffleSplit(target, n_iter=5, test_size=0.5, random_state=42)
random_search = RandomizedSearchCV(clf,
                                   param_dist,
                                   random_state=random_state,
                                   cv=skf,
                                   scoring='log_loss',
                                   verbose=3,
                                   n_jobs=1,
                                   n_iter=100)

# cv = StratifiedShuffleSplit(target, n_iter=5, test_size=0.3, random_state=42)
# grid = GridSearchCV(SVC(), param_grid=param_dist, cv=cv, n_jobs=1, verbose=3)
#
fit = random_search.fit(X, target)
report(fit.grid_scores_)