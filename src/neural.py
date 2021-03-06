#!/usr/bin/env python
from __future__ import division
from sklearn.ensemble import RandomForestClassifier

__author__ = 'Vladimir Iglovikov'

#import libraries
import pandas as pd
import os
import numpy as np
import random
import math
from sklearn.preprocessing import scale
import sys

sys.path += [os.path.join("/home/vladimir/workspace/NeuralNetworks/multilayer_perceptron")]
sys.path += [os.path.join("/home/vladimir/workspace/NeuralNetworks")]
from multilayer_perceptron  import MultilayerPerceptronClassifier

random.seed(666)

data_path = os.path.join("..", "data")

training = pd.read_csv(os.path.join(data_path, "train.csv"))
testing = pd.read_csv(os.path.join(data_path, "test.csv"))

target = training["target"].values

train = training.drop(["target", "id"], axis=1)

def fun(x):
    return math.log(x + 1)

train_log = train.applymap(fun)
train_log_scale = scale(train_log.values.astype(float), axis=0)


id = testing["id"]

testing = testing.drop(["id"], axis=1)

# testing_log = testing.applymap(fun)

# testing_log_scale = scale(testing_log.values.astype(float), axis=0)

clf = MultilayerPerceptronClassifier(hidden_layer_sizes=(200, 100), max_iter = 200, alpha = 0.02)

print "Training"
# fit = forest.fit(train_log_scale, target)
fit = clf.fit(train, target)

def predicting(testing, n):
    '''
    There are problems with memory that I have when I am trying to predict all testing set
    at a time. But I believe I can split testing set, predict on each part and merge after this.

    :param testing: set that needs to be predicted
    :param n
    :return:
    '''
    frames = np.array_split(testing, n)

    prediction = map(fit.predict_proba, frames)

    return np.concatenate(prediction)


print "Predicting"
prediction = fit.predict_proba(testing.values)
# prediction = predicting(testing_log_scale, 10)


temp = pd.DataFrame(prediction)

temp = temp.rename(columns = {0: "Class_1",
                              1: "Class_2",
                              2: "Class_3",
                              3: "Class_4",
                              4: "Class_5",
                              5: "Class_6",
                              6: "Class_7",
                              7: "Class_8",
                              8: "Class_9"})

temp["id"] = id

temp.to_csv(os.path.join("..", "data", "NN_200_100.csv"), index=False)