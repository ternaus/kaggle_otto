from __future__ import division
__author__ = 'Vladimir Iglovikov'

'''
This script reads the data preforms PCA, and keeps only N of the best components
'''

from sklearn import decomposition
import pandas as pd
import os

N = 100
print "reading train"
train = pd.read_csv(os.path.join("..", "data", "train_squared.csv"))

print "reading test"
test = pd.read_csv(os.path.join("..", "data", "test_squared.csv"))

target = train["target"]
train_id = train["id"]
test_id = test["id"]

train = train.drop(["id", "target"], 1)
test = test.drop("id", 1)

pca = decomposition.PCA(n_components=N)
print "fitting"
pca.fit(train)
print "transforming train"
x = pd.DataFrame(pca.transform(train))

x["id"] = train_id
x["target"] = target

x.to_csv(os.path.join("..", "data", "train_pca_{N}".format(N=N)))

print "transforming test"
y = pd.DataFrame(pca.transform(test))
y["id"] = test_id
y.to_csv(os.path.join("..", "data", "test_pca_{N}".format(N=N)))


