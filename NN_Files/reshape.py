import sys
import numpy as np
import Utils as U
import pandas as pd

# Reshape output from LSTM for calibration analysis

paths = U.load_paths()
pred_probs_test = pd.read_csv('LSTM_test_pred_probscalibration.csv')
pred_probs_train = pd.read_csv('LSTM_train_pred_probscalibration.csv')

og_train = pd.read_csv(paths['data_dir'] + 'train_ts.csv')
og_test = pd.read_csv(paths['data_dir'] + 'test_ts.csv')

test_ids = pd.read_csv("unscaled_test_ts.csv").iloc[:, 1].astype(int)

test_new = pd.read_csv("data/test_new.csv")

test_out = pred_probs_test.to_numpy().reshape(og_test.shape[0], -1)
train_out = pred_probs_train.to_numpy().reshape(og_train.shape[0], -1)
test_ids_out = test_ids.to_numpy().reshape(og_test.shape[0], -1)


test_samples = np.mean(test_out, 1)[0::24]
train_samples = np.mean(train_out, 1)[0::24]
test_ids_samples = np.mean(test_ids_out, 1)[0::24]

test_and_ids = pd.DataFrame(np.stack((test_samples, test_ids_samples)).T)
test_and_ids.loc[:, 1] = test_and_ids.loc[:, 1].astype(int)
merged = pd.merge(test_new, test_and_ids, how = 'left', left_on = 'patient', right_on = 1)
merged = merged.drop_duplicates()
merged[0].to_csv("lstm_predicted_probs_12_12.csv", index = False)