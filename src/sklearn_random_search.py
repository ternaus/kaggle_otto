#!/usr/bin/env python
__author__ = 'Vladimir Iglovikov'

'''
This script will do randomized search to find the best or almost the best parameters for this problem
for sklearn package
'''

import xgboost as xgb
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn import cross_validation
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedKFold
import math
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

target = train["target"].values
training = train.drop(["id", 'target'], 1).values
testing = test.drop("id", 1)

clf = RandomForestClassifier()

from operator import itemgetter
# Utility function to report best scores
def report(grid_scores, n_top=5):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
from scipy.stats import randint as sp_randint

param_dist = {"n_estimators": [10, 20, 100],
              "max_depth": [3, 4, 10, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [False, True]}

random_search = RandomizedSearchCV(clf, param_dist, n_jobs=-1, random_state=42, cv=5, scoring='log_loss')
fit = random_search.fit(training, target)
report(fit.grid_scores_)