#!/usr/bin/env python
from __future__ import division
from sklearn.ensemble import RandomForestClassifier

__author__ = 'Vladimir Iglovikov'

#import libraries
import pandas as pd
import os
import numpy as np
import random

random.seed(666)

data_path = os.path.join("..", "data")

training = pd.read_csv(os.path.join(data_path, "train.csv"))
testing = pd.read_csv(os.path.join(data_path, "test.csv"))

target = training["target"].values

train = training.drop(["target", "id"], axis=1)
testing = testing.drop(["id"], axis=1)

forest = RandomForestClassifier(n_estimators=1000, n_jobs=3)

fit = forest.fit(train, target)

prediction = fit.predict(testing.values)

submission = pd.read_csv(os.path.join(data_path, "sampleSubmission.csv"))
num_rows = len(prediction)

for item in set(prediction.tolist()):
    submission[item] = [0] * num_rows

for i in range(len(prediction)):
    item = prediction[i]
    submission[item][i] = 1

submission.to_csv(os.path.join("..", "data", "rf_1000.csv"), index=False)