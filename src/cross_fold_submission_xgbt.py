#!/usr/bin/env python
from __future__ import division
__author__ = 'Vladimir Iglovikov'

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import pandas as pd
import graphlab as gl
import sys
sys.path += ["/home/vladimir/compile/xgboost/wrapper"]

import xgboost as xgb
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn import cross_validation
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedKFold
import math

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

target = train["target"].values
training = train.drop(["id", 'target'], 1).values
testing = test.drop("id", 1)

skf = cross_validation.StratifiedKFold(target, n_folds=10, random_state=42)

# params = {'n_estimators': 1000,
#           'n_jobs': -1}

def make_submission(m, test, filename):
    preds = m.predict_topk(test, output_type='probability', k=9)
    preds['id'] = preds['id'].astype(int) + 1
    preds = preds.unstack(['class', 'probability'], 'probs').unpack('probs', '')
    preds = preds.sort('id')
    preds.save(filename)

def multiclass_logloss(model, test):
    preds = model.predict_topk(test, output_type='probability', k=9)
    preds = preds.unstack(['class', 'probability'], 'probs').unpack('probs', '')
    preds['id'] = preds['id'].astype(int) + 1
    preds = preds.sort('id')
    preds['target'] = test['target']
    neg_log_loss = 0
    for row in preds:
        label = row['target']
        neg_log_loss += - math.log(row[label])
    return neg_log_loss / preds.num_rows()

def evaluate_logloss(model, train, valid):
    return {'train_logloss': multiclass_logloss(model, train),
            'valid_logloss': multiclass_logloss(model, valid)}

result = []

ind = 1
for train_index, test_index in skf:
    X_train, X_test = training[train_index], target[train_index]
    temp = gl.SFrame(X_train)

    temp["target"] = X_test

    params = {'target': 'target',
          'max_iterations': 250,
          'max_depth': 10,
          'min_child_weight': 4,
          'row_subsample': 0.9,
          'min_loss_reduction': 1,
          'column_subsample': 0.8,
          # 'features' : features,
          'validation_set': None}


    clf = gl.boosted_trees_classifier.create(temp, **params)
    temp1 = gl.SFrame(training[test_index])
    temp1['target'] = target[test_index]
    a = evaluate_logloss(clf, temp, temp1)
    print a
    result += [a['valid_logloss']]
    # fit = clf.fit(X_train, X_test)
    # prediction_1 = fit.predict_proba(training[test_index])
    # print log_loss(target[test_index], prediction_1)
    # prediction_2 = fit.predict_proba(testing.values)
    # submission = pd.DataFrame(prediction_2)
    # submission.columns = ["Class_" + str(i) for i in range(1, 10)]
    # submission["id"] = test["id"]
    make_submission(clf, gl.SFrame(testing.values), "btc_mi_250_md10_mch4_rs09_mlr1_ind{ind}.csv".format(ind=ind))
    # submission.to_csv("btc_mi_250_md10_mch4_rs09_mlr1_ind{ind}.csv".format(ind=ind), index=False)
    ind += 1

print result
print np.mean(result)