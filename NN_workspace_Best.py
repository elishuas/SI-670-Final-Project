# -*- coding: utf-8 -*-
# # Neural Net Training Workspace
# *Work produced by Stephen Toner, Fall 2022*

import torch
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

#params = {'legend.fontsize': 'x-large',
#         'axes.labelsize': 'x-large',
#         'axes.titlesize':'x-large',
#        'xtick.labelsize':'x-large',
#         'ytick.labelsize':'x-large'}
#pylab.rcParams.update(params)

# +
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
# -

# +

data = pd.read_csv("timeseries_readyformodel.csv")

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

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
# - 

# +
net = models.Sequential()
net.add(layers.Dense(16, activation = 'tanh', input_shape =(None, X_train_scaled.shape[0], X_train_scaled.shape[1])))
net.add(layers.Dense(32, activation = 'tanh'))
net.add(layers.Dense(64, activation = 'tanh')) # Up to 82%
# net.add(layers.LSTM(64, activation = 'tanh'))
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
net.fit(X_train_scaled, y_train_cat, epochs = 1000, batch_size = 200)#, verbose = False)
# -

# +
test_accuracy = net.evaluate(X_test_scaled, y_test_cat)[1]
# -


print(f'Test Accuracy: {test_accuracy}')
# feval_metrics = U.evaluation_metrics(net, X_train_scaled, y_train_cat, X_test_scaled, y_test_cat)

# y_pred_prob = net.predict_prob()
# results = U.compute_metrics(y_pred_prob, y_test)