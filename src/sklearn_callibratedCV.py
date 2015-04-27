
#!/usr/bin/env python
from __future__ import division
__author__ = 'Vladimir Iglovikov'

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
import os
import cPickle as pickle
import gzip

print 'reading train'
train = pd.read_csv('../data/train.csv')

print 'reading test'
test = pd.read_csv('../data/test.csv')

target = train["target"].values
training = train.drop(["id", 'target'], 1).values
testing = test.drop("id", 1)


random_state = 42

n_folds = 10



# model = 'rf' #RandomForest
model = 'gb' #GradientBoosting

if model == 'rf':
    params = {'n_estimators': 100,
              'n_jobs': -1,
              'random_state': random_state}
    method = 'rf_{n_estimators}_nfolds_{n_folds}'.format(n_folds=n_folds, n_estimators=params['n_estimators'])
    clf = RandomForestClassifier(**params)
elif model == 'gb':
    params = {'n_estimators': 1000,
              'random_state': random_state}
    method = 'gb_{n_estimators}_nfolds_{n_folds}'.format(n_folds=n_folds, n_estimators=params['n_estimators'])
    clf = GradientBoostingClassifier(**params)

skf = cross_validation.StratifiedKFold(target, n_folds=n_folds, random_state=random_state)

ccv = CalibratedClassifierCV(base_estimator=clf, method='sigmoid', cv=skf)

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