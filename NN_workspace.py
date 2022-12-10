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

y = data['died']
X = data.drop(columns = ['died', 'patientunitstayid']).to_numpy().astype('float32')


# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    # stratify=X[['hospitalregion',
                                                    #             'teachingstatus']],
                                                    random_state=607)

# -

# +
scaler = StandardScaler()
# scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# X_train_scaled = X_train
# X_test_scaled = X_test

# y_train_cat = to_categorical(y_train)
# y_test_cat = to_categorical(y_test)

y_train_cat = y_train
y_test_cat = y_test
# - 

# +
X_tr_ts = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_tst_ts = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))


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
net.add(layers.Dense(2, activation = 'sigmoid'))

# -

# +
net.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [AUC(),Recall(), Precision() ])
net.fit(X_tr_ts, y_train_cat, epochs = 100, batch_size = 200)#, verbose = False)
# -

# +
test_accuracy = net.evaluate(X_tst_ts, y_test_cat)[1]
# -




net.save(model_dir + 'LSTM_model' + suffix)


print(f'Test Accuracy: {test_accuracy}')
print(f'Test Accuracy: {test_accuracy}')
y_pred_prob = net.predict(X_test_scaled)

pd.DataFrame(y_pred_prob).to_csv(paths['prediction_dir'] + "LSTM" + suffix + ".csv", index = False)