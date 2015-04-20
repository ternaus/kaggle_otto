#!/usr/bin/env python
from __future__ import division
__author__ = 'Vladimir Iglovikov'

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import pandas as pd



train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

target = train["target"].values
training = train.drop(["id", 'target'], 1).values
testing = test.drop("id", 1)

skf = cross_validation.StratifiedKFold(target, n_folds=10, random_state=42)

params = {'n_estimators': 1000,
          'n_jobs': -1}

ind = 1
for train_index, test_index in skf:
    X_train, X_test = training[train_index], target[train_index]

    clf = RandomForestClassifier(**params)
    fit = clf.fit(X_train, X_test)
    prediction_1 = fit.predict_proba(training[test_index])
    print log_loss(target[test_index], prediction_1)
    prediction_2 = fit.predict_proba(testing.values)
    submission = pd.DataFrame(prediction_2)
    submission.columns = ["Class_" + str(i) for i in range(1, 10)]
    submission["id"] = test["id"]
    submission.to_csv("rf_1000_cv10_ind{ind}.csv".format(ind=ind), index=False)
    ind += 1
