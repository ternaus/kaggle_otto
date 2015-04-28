from __future__ import division
__author__ = 'Vladimir Iglovikov'

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
import os
import cPickle as pickle
import gzip
import xgboost
import gl_wrapper
print 'reading train'
train = pd.read_csv('../data/train.csv')

print 'reading test'
test = pd.read_csv('../data/test.csv')

target = train["target"].values
training = train.drop(["id", 'target'], 1).values
testing = test.drop("id", 1)


random_state = 42

n_folds = 10

calibration_method = 'isotonic'

# model = 'rf' #RandomForest
# model = 'gb' #GradientBoosting
# model = 'xgb' #eXtremeGradient Boosting
#model = 'xgbt'
model = 'svm'

if model == 'rf':
    params = {'n_estimators': 100,
              'n_jobs': -1,
              'random_state': random_state}
    method = 'rf_{n_estimators}_nfolds_{n_folds}_calibration_{calibration_method}'.format(n_folds=n_folds, n_estimators=params['n_estimators'], calibration_method=calibration_method)
    clf = RandomForestClassifier(**params)
elif model == 'gb':
    params = {'n_estimators': 1000,
              'random_state': random_state}
    method = 'gb_{n_estimators}_nfolds_{n_folds}_calibration_{calibration_method}'.format(n_folds=n_folds, n_estimators=params['n_estimators'], calibration_method=calibration_method)
    clf = GradientBoostingClassifier(**params)
elif model == 'xgb':
    params = {'max_depth': 10,
                    'n_estimators': 100}

    method = 'xgb_{n_estimators}_md{md}_nfolds_{n_folds}_calibration_{calibration_method}'.format(md=params['max_depth'],
                                                                                                  n_folds=n_folds,
                                                                                                  n_estimators=params['n_estimators'],
                                                                                                  calibration_method=calibration_method)
    clf = xgboost.XGBClassifier(**params)
elif model == 'xgbt':
    params = {'max_iterations': 300, 'max_depth': 8, 'min_child_weight': 4, 'row_subsample': 0.9, 'min_loss_reduction': 1, 'column_subsample': 0.8}
    method = 'xgbt_{max_iterations}_max_depth{max_depth}_min_loss_reduction{min_loss_reduction}_min_child_weight{min_child_weight}_row_subsample{row_subsample}_column_subsample{column_subsample}_nfolds_{n_folds}_calibration_{calibration_method}'.format(max_depth=params['max_depth'],
                                                                                                  max_iterations=params['max_iterations'],
                                                                                                  min_loss_reduction=params['min_loss_reduction'],
                                                                                                  min_child_weight=params['min_child_weight'],
                                                                                                  row_subsample=params['row_subsample'],
                                                                                                  column_subsample=params['column_subsample'],
                                                                                                  calibration_method=calibration_method,
                                                                                                  n_folds=n_folds)
    clf = gl_wrapper.BoostedTreesClassifier(**params)

elif model == 'svm':
    params = {'C': 1}
    method = 'svm_{C}_nfolds_{n_folds}_calibration_{calibration_method}'.format(n_folds=n_folds,
                                                                                C=params['C'],
                                                                                calibration_method=calibration_method)
    clf = OneVsRestClassifier(SVC(**params), n_jobs=-1)


skf = cross_validation.StratifiedKFold(target, n_folds=n_folds, random_state=random_state)

ccv = CalibratedClassifierCV(base_estimator=clf, method=calibration_method, cv=skf)

print 'fit the data'

fit = ccv.fit(training, target)

print 'predict on training set'
score = log_loss(target, fit.predict_proba(training))
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
prediction = ccv.predict_proba(testing.values)
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