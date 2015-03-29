from __future__ import division
__author__ = 'Vladimir Iglovikov'


import graphlab as gl
import math
import random
random.seed(666)
import os
import time

data_path = os.path.join("..", "data")
train = gl.SFrame.read_csv(os.path.join(data_path, 'train.csv'))
# test = gl.SFrame.read_csv('data/test.csv')
del train['id']


job = gl.toolkits.model_parameter_search(gl.boosted_trees_classifier.create,
                                 training_set=train,
                                 target='target',
                                 # max_iterations=[10],
                                 max_depth=[6, 7, 8, 9, 10],
                                 min_child_weight=[1, 2, 3, 4, 5, 6],
                                 min_loss_reduction=[1, 2, 3]
                                 )

job_result = job.get_results()

summary_sframe = job_result['summary']

# print summary_sframe
result = open("report_{timestamp}.txt".format(timestamp=time.time()), "w")
print  >> result, summary_sframe
result.close()
