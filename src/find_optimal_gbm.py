from __future__ import division
__author__ = 'Vladimir Iglovikov'

import time
import pandas as pd
import os
import numpy as np
import random
import math
from sklearn.preprocessing import scale
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split

data_path = os.path.join("..", "data")
training = pd.read_csv(os.path.join(data_path, "train.csv"))
# testing = pd.read_csv(os.path.join(data_path, "test.csv"))

target = training["target"].values

train = training.drop(["target", "id"], axis=1)

gbdt = GradientBoostingClassifier()

tuned_parameters = [{'n_estimators': [1]}]

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.3, random_state=0)
clf = GridSearchCV(gbdt, tuned_parameters, cv=5, scoring="log_loss", n_jobs=-1)
clf.fit(X_train, y_train)



fName = open("report_{timestamp}.txt".format(timestamp=time.time()), 'w')
print >> fName, "Best parameters set found on development set:"
print >> fName, clf.best_estimator_
print >> fName, "Grid scores on development set:"

for params, mean_score, scores in clf.grid_scores_:
    fName.write("%0.3f (+/-%0.03f) for %r"
          % (mean_score, scores.std() / 2, params))


fName.write("Detailed classification report:\n")

fName.write("The model is trained on the full development set.\n")
fName.write("The scores are computed on the full evaluation set.\n")

y_true, y_pred = y_test, clf.predict(X_test)
print >> fName, classification_report(y_true, y_pred)

fName.close()