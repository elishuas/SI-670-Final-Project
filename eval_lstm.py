import sys
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras import models
from keras import layers
from tensorflow.keras.utils import to_categorical
import Utils as U

from tensorflow.keras.models import load_model
import pandas as pd

from sklearn.model_selection import train_test_split


if len(sys.argv) > 1:
    suffix = sys.argv[1]
else:
    suffix = "latest"

paths = U.load_paths()

data_dir = paths['data_dir']
model_dir = paths['models_path']

data = pd.read_csv(data_dir + "timeseries_readyformodel.csv")


train = pd.read_csv(data_dir + 'train_ts.csv')
test = pd.read_csv(data_dir + 'test_ts.csv')

train_patient_ids = train['patient']
test_patient_ids = test['patient']

train_ts = pd.merge(train_patient_ids, data, how = "left", on = 'patient')
test_ts = pd.merge(test_patient_ids, data, how = "left", on = 'patient')


X_train = train_ts.drop(columns = "died")
y_train = train_ts['died']
X_test = test_ts.drop(columns = "died")
y_test = test_ts['died']

scaler = StandardScaler()
# scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# +
# def to_timeseries(X):
#     return X.reshape((X.shape[0], 1, X.shape[1]))

X_tr_ts = U.to_timeseries(X_train_scaled)
X_tst_ts =  U.to_timeseries(X_test_scaled)

model = load_model(paths['models_path'] + "LSTM_model_12_11tiny")

model.summary()

for layer in model.layers:
    print(layer.input_shape)
# results = model.evaluate(X_tst_ts, y_test)


train_pred_probs = model.predict(X_tr_ts)
test_pred_probs = model.predict(X_tst_ts)

pd.DataFrame(test_pred_probs).to_csv("LSTM_test_pred_probs" + suffix + ".csv", index = False)
pd.DataFrame(train_pred_probs).to_csv("LSTM_train_pred_probs" + suffix + ".csv", index = False)

train_results = U.compute_metrics(train_pred_probs, y_train)
test_results = U.compute_metrics(test_pred_probs, y_test)

train_precision, train_recall, _ = (
    U.precision_recall_curve(y_train, 
                           model.predict(X_tr_ts))
)
test_precision, test_recall, _ = (
    U.precision_recall_curve(y_test, 
                           model.predict(X_tst_ts))
)

train_auprc = U.auc(train_recall, train_precision)
test_auprc = U.auc(test_recall, test_precision)

print(train_results)
print(test_results)
print(train_auprc)
print(test_auprc)

U.save_metrics({"train": train_results, "test": test_results},
                [train_auprc, test_auprc],
                model_name ="LSTM_Tiny_calibration",
                save = True)
