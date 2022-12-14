# -*- coding: utf-8 -*-
# # Neural Net Training Workspace
# *Work produced by Stephen Toner, Fall 2022*
import sys
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras import models
from keras import layers
from tensorflow.keras.utils import to_categorical
import pandas as pd
from tensorflow.keras.metrics import AUC, Recall, Precision 
from sklearn.model_selection import train_test_split



import Utils as U

# +

np.random.seed(0)
random.seed(0)
# -

# +

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

# -

# +
net = models.Sequential()
net.add(layers.LSTM(24, activation = 'tanh', input_shape = (X_tr_ts.shape[1], X_tr_ts.shape[2])))


# Output Layer
net.add(layers.Dense(1, activation = 'sigmoid'))

# -

# +
net.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [AUC(),Recall(), Precision() ])
net.fit(X_tr_ts, y_train)#, verbose = False)
# -

# +
test_accuracy = net.evaluate(X_tst_ts, y_test)[1]
# -

net.save('LSTM_model_12_11_tiny' + suffix)


# print(f'Test Accuracy: {test_accuracy}')
# feval_metrics = U.evaluation_metrics(net, X_train_scaled, y_train_cat, X_test_scaled, y_test_cat)

# y_pred_prob = net.predict_prob()
# results = U.compute_metrics(y_pred_prob, y_test)