from __future__ import division

__author__ = 'Vladimir Iglovikov'

#import libraries
import pandas as pd
import os
import numpy as np
data_path = os.path.join("..", "data")

training = pd.read_csv(os.path.join(data_path, "train.csv"))

train = training.drop(["target", "id"], axis=1)

def add_squared(df):
    N = len(df.columns)
    for i in range(N):
        column_i = df.columns[i]
        for j in range(i, N):
            column_j = df.columns[j]
            df[column_i + column_j] = df[column_i] * df[column_j]
    return df

# frames = np.array_split(test.values, 10)

# frames = [pd.DataFrame(x) for x in frames]

# for frame in frames:
#     frame.columns = test.columns

# result = map(add_squared, frames)

# test = pd.concat(result)

train = add_squared(train)
train["id"] = training["id"]
train["target"] = training["target"]
train.to_csv(os.path.join(data_path, "train_squared.csv"))
