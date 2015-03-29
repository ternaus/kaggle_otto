from __future__ import division

__author__ = 'Vladimir Iglovikov'

#import libraries
import pandas as pd
import os
import numpy as np
data_path = os.path.join("..", "data")

testing = pd.read_csv(os.path.join(data_path, "test.csv"))

test = testing.drop("id", axis=1)

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

test = add_squared(test)
test["id"] = testing["id"]

test.to_csv(os.path.join(data_path, "test_squared.csv"))
