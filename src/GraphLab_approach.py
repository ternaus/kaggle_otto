__author__ = 'Vladimir Iglovikov'

import graphlab as gl
import math
import random
import os

random.seed(666)
data_path = os.path.join("..", "data")

train = gl.SFrame.read_csv(os.path.join(data_path, "train.csv"))
test = gl.SFrame.read_csv(os.path.join(data_path, "test.csv"))

del train['id']

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

def shuffle(sf):
    sf['_id'] = [random.random() for i in xrange(sf.num_rows())]
    sf = sf.sort('_id')
    del sf['_id']
    return sf

def evaluate_logloss(model, train, valid):
    return {'train_logloss': multiclass_logloss(model, train),
            'valid_logloss': multiclass_logloss(model, valid)}

params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 8,
          'min_child_weight': 4,
          'row_subsample': 0.9,
          'min_loss_reduction': 2,
          'column_subsample': 0.9,
          'validation_set': None}

# params = {'target': 'target',
#           'validation_set': None,
#           'max_iterations': 250,
#           'min_loss_redution': 1}


train = shuffle(train)

# Check performance on internal validation set
tr, va = train.random_split(.8)
m = gl.boosted_trees_classifier.create(tr, **params)
print evaluate_logloss(m, tr, va)

# Make final submission by using full training set
m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, os.path.join(data_path, 'xbt_mi300_md_8_mcw_4_rs_0.9_mlr2_cs0.9.csv'))
# make_submission(m, test, os.path.join(data_path, 'xbt_mi500mcw_4.csv'))
# make_submission(m, test, os.path.join(data_path, 'xbt_default.csv'))