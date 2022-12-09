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

# Create function to compute optimal F-beta score using threshold tuning
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


def compute_area_under_precision_recall_curve(y_pred_prob, y_true):
    
    precision, recall, _ = precision_recall_curve(y_train, y_pred_prob)
    auprc = auc(recall, precision)
    return auprc

def bootstrap_model_metrics(model, train_data, test_data, target_col, 
                            n_boot_samples):
    
    n_samples_train = train_data.shape[0]
    n_samples_test = test_data.shape[0]
    
    train_auprc_boot_samples = np.zeros((n_boot_samples,))
    train_precision_boot_samples = np.zeros((n_boot_samples,))
    train_recall_boot_samples = np.zeros((n_boot_samples,))
    train_fscore_boot_samples = np.zeros((n_boot_samples,))
    
    test_auprc_boot_samples = np.zeros((n_boot_samples,))
    test_precision_boot_samples = np.zeros((n_boot_samples,))
    test_recall_boot_samples = np.zeros((n_boot_samples,))
    test_fscore_boot_samples = np.zeros((n_boot_samples,))
    
    for boot_iter in range(n_boot_samples):
        # Generate bootstrap samples with replacement from training and test 
        # sets
        boot_train_data = train_data[np.random.choice(n_samples_train, 
                                                      size=n_samples_train, 
                                                      replace=True), :]
        boot_test_data = test_data[np.random.choice(n_samples_test, 
                                                    size=n_samples_test, 
                                                    replace=True), :]
        
        X_boot_train = boot_train.drop(columns=target_col)
        X_boot_test = boot_test.drop(columns=target_col)
        y_boot_train = boot_train[target_col]
        y_boot_test = boot_test[target_col]
        
        model.fit(X_boot_train, y_boot_train)
        
        boot_train_pred_probs = model.predict_proba(X_boot_train)
        boot_test_pred_probs = model.predict_proba(X_boot_test)
        
        boot_train_results = compute_metrics(boot_train_pred_probs, 
                                             y_boot_train)
        boot_test_results = compute_metrics(boot_test_pred_probs, 
                                             y_boot_test)
        
        # Compute metrics on training set
        train_auprc_boot_samples[boot_iter] = (
            compute_area_under_precision_recall_curve(boot_train_pred_probs, 
                                                      y_boot_train)
        )
        train_precision_boot_samples[boot_iter] = (
            boot_train_results['precision'].values[0]
        )
        train_recall_boot_samples[boot_iter] = (
            boot_train_results['recall'].values[0]
        )
        train_fscore_boot_samples[boot_iter] = (
            boot_train_results['f-score'].values[0]
        )
        
        # Compute metrics on test set
        test_auprc_boot_samples[boot_iter] = (
            compute_area_under_precision_recall_curve(boot_test_pred_probs, 
                                                      y_boot_test)
        )
        test_precision_boot_samples[boot_iter] = (
            boot_test_results['precision'].values[0]
        )
        test_recall_boot_samples[boot_iter] = (
            boot_test_results['recall'].values[0]
        )
        test_fscore_boot_samples[boot_iter] = (
            boot_test_results['f-score'].values[0]
        )
        
    train_auprc_lower = np.quantile(train_auprc_boot_samples, 0.025)
    train_auprc_upper = np.quantile(train_auprc_boot_samples, 0.975)
    train_precision_lower = np.quantile(train_precision_boot_samples, 0.025)
    train_precision_upper = np.quantile(train_precision_boot_samples, 0.975)
    train_recall_lower = np.quantile(train_recall_boot_samples, 0.025)
    train_recall_upper = np.quantile(train_recall_boot_samples, 0.975)
    train_fscore_lower = np.quantile(train_fscore_boot_samples, 0.025)
    train_fscore_upper = np.quantile(train_fscore_boot_samples, 0.975)
    
    test_auprc_lower = np.quantile(test_auprc_boot_samples, 0.025)
    test_auprc_upper = np.quantile(test_auprc_boot_samples, 0.975)
    test_precision_lower = np.quantile(test_precision_boot_samples, 0.025)
    test_precision_upper = np.quantile(test_precision_boot_samples, 0.975)
    test_recall_lower = np.quantile(test_recall_boot_samples, 0.025)
    test_recall_upper = np.quantile(test_recall_boot_samples, 0.975)
    test_fscore_lower = np.quantile(test_fscore_boot_samples, 0.025)
    test_fscore_upper = np.quantile(test_fscore_boot_samples, 0.975)
        
    print(f'95% CI for train AUPRC: ({train_auprc_lower}, {train_auprc_upper})')
    print(f'95% CI for train precision: ({train_precision_lower}, {train_precision_upper})')
    print(f'95% CI for train recall: ({train_recall_lower}, {train_recall_upper})')
    print(f'95% CI for train F-score: ({train_fscore_lower}, {train_fscore_upper}) \n')
    
    print(f'95% CI for test AUPRC: ({test_auprc_lower}, {test_auprc_upper})')
    print(f'95% CI for test precision: ({test_precision_lower}, {test_precision_upper})')
    print(f'95% CI for test recall: ({test_recall_lower}, {test_recall_upper})')
    print(f'95% CI for test F-score: ({test_fscore_lower}, {test_fscore_upper})')

def plot_calibration_curve(labels, pred_prob, plot_model_label):
    
    x, y = calibration_curve(labels, pred_prob, n_bins = 10)
     
    # Plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated')
     
    # Plot model's calibration curve
    plt.plot(y, x, marker = '.', label = plot_model_label)
     
    leg = plt.legend(loc = 'lower right')
    plt.xlabel('Average predicted probability in each bin')
    plt.ylabel('Fraction of deceased patients')
    plt.show()