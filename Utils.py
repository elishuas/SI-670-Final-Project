import numpy as np
import pandas as pd

from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def evaluation_metrics(model, X_train, y_train, X_test, y_test):
    train_precision, train_recall, _ = (
        precision_recall_curve(y_train, model.predict_proba(X_train))
    )
    test_precision, test_recall, _ = (
        precision_recall_curve(y_test, model.predict_proba(X_test))
    )

    train_auprc = auc(train_recall, train_precision)
    test_auprc = auc(test_recall, test_precision)

    results = [train_precision, 
            train_recall, test_precision, test_recall, train_auprc, test_auprc]
    labels = ["train_precision", 
            "train_recall", "test_precision", "test_recall", "train_auprc", "test_auprc"]

    return {l:r for (l, r) in list(zip(labels, results))}

def compute_metrics(y_pred_prob, y_true):
    
    thresholds = np.arange(0, 1.01, 0.01)
    results = pd.DataFrame(columns=['threshold', 'precision', 'recall', 
                                    'f-score'])
    
    for threshold in thresholds:
        y_pred = np.where(y_pred_prob >= threshold, 1, 0)
          
        # Compute precision
        precision = precision_score(y_true, y_pred)
        
        # Compute recall
        recall = recall_score(y_true, y_pred)
        
        # Compute F-beta score
        f_score = fbeta_score(y_true, y_pred, beta=2)
        
        results = results.append({'threshold': threshold,
                                  'precision': precision,
                                  'recall': recall,
                                  'f-score': f_score}, 
                                 ignore_index=True)
        
    return results.loc[results['f-score'] == results['f-score'].max(), :]

import os
import csv
import pickle
import json


def write_pkl(data, path, verbose=1):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    if verbose:
        print('saved to ', path)

def write_json(data, path, sort_keys=False, verbose=1):
    with open(path, 'w') as outfile:
        json.dump(data, outfile, sort_keys=sort_keys, indent=4)
    if verbose:
        print('saved to ', path)


def load_json(path):
    with open(path, 'r') as json_file:
        info = json.load(json_file)
    return info

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def record_results(path_results, hparams, log_dict):
    results = {r: log_dict[r] for r in log_dict if r != 'test_conf_m'}
    header = [h for h in hparams] + [r for r in results]
    ret = {**hparams, **results}
    file_exists = os.path.isfile(path_results)
    with open(path_results, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(ret)
    print('Written results at ', path_results)
