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

# def create_dataset(dataset, look_back=5):
#     dataX, dataY = [], []
#     for i in range(len(dataset)-look_back-1):
#         a = dataset[i:(i+look_back), 0]
#     dataX.append(a)
#     dataY.append(dataset[i + look_back, 0])
#     return np.array(dataX), np.array(dataY)

#params = {'legend.fontsize': 'x-large',
#         'axes.labelsize': 'x-large',
#         'axes.titlesize':'x-large',
#        'xtick.labelsize':'x-large',
#         'ytick.labelsize':'x-large'}
#pylab.rcParams.update(params)

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

train_ts = pd.merge(train_patient_ids, data, on = 'patient')
test_ts = pd.merge(test_patient_ids, data, on = 'patient')


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

# X_tr_ts = X_train_scaled
# X_tst_ts = X_test_scaled
# -

# +
net = models.Sequential()
net.add(layers.LSTM(1000, activation = 'tanh', input_shape = (X_tr_ts.shape[1], X_tr_ts.shape[2])))
net.add(layers.Dense(16, activation = 'tanh'))
net.add(layers.Dense(32, activation = 'tanh'))
net.add(layers.Dense(64, activation = 'tanh')) # Up to 82%

# net.add(layers.Dense(128, activation = 'tanh')) # Up to 82.5%
# net.add(layers.Dense(256, activation = 'tanh')) # 
# net.add(layers.Dense(512, activation = 'tanh')) # 
# # net.add(layers.Conv1D(256, kernel_size = 2))# 
net.add(layers.Dense(256, activation = 'relu')) # 85% for flat 256, 3 tanh layers; down to 84% when increasing powers of 2
# # net.add(layers.Dense(32, activation = 'relu'))
# # net.add(layers.Dense(16, activation = 'relu'))
# net.add(layers.Dense(8, activation = 'relu'))

# Output Layer
net.add(layers.Dense(1, activation = 'sigmoid'))

# -

# +
net.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [AUC(),Recall(), Precision() ])
net.fit(X_tr_ts, y_train, epochs = 1000, batch_size = 200)#, verbose = False)
# -

# +
test_accuracy = net.evaluate(X_tst_ts, y_test)[1]
# -

net.save('LSTM_model_12_11')


# print(f'Test Accuracy: {test_accuracy}')
# feval_metrics = U.evaluation_metrics(net, X_train_scaled, y_train_cat, X_test_scaled, y_test_cat)

# y_pred_prob = net.predict_prob()
# results = U.compute_metrics(y_pred_prob, y_test)