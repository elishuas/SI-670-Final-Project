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

train = pd.read_csv(data_dir + 'train_new.csv')
test = pd.read_csv(data_dir + 'test_new.csv')
data = pd.concat([train, test])


y_train = np.asarray(train['died']).astype('float32').reshape((-1,1))
X_train = train.drop(columns = ['died']).to_numpy().astype('float32')


y_test = np.asarray(test['died']).astype('float32').reshape((-1,1))
X_test = test.drop(columns = ['died']).to_numpy().astype('float32')
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
# - 

# +
net = models.Sequential()
net.add(layers.Dense(128, activation = 'tanh', input_shape =(None, X_train_scaled.shape[0], X_train_scaled.shape[1])))
net.add(layers.Dense(32, activation = 'tanh'))
net.add(layers.Dropout(0.2))
net.add(layers.Dense(64, activation = 'tanh')) # Up to 82%
# net.add(layers.LSTM(64, activation = 'tanh'))
# net.add(layers.Dense(128, activation = 'relu')) # Up to 82.5%
# net.add(layers.Dense(256, activation = 'relu')) # 
# net.add(layers.Dense(512, activation = 'relu')) # 
# # # net.add(layers.Conv1D(256, kernel_size = 2))# 
net.add(layers.Dropout(0.2))
net.add(layers.Dense(16, activation = 'tanh')) # Up to 82%
# net.add(layers.Dense(16, activation = 'relu'))
# net.add(layers.Dense(8, activation = 'relu'))

# Output Layer
net.add(layers.Dense(1, activation = 'sigmoid'))

# -

# +
net.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [AUC(),Recall(), Precision() ])
net.fit(X_train_scaled, y_train, epochs = 100, batch_size = 200)#, verbose = False)
# -

# +
test_accuracy = net.evaluate(X_test_scaled, y_test)
# -


net.save(model_dir + 'MLP_model' +  suffix)

print(f'Test Accuracy: {test_accuracy}')
y_pred_prob = net.predict(X_test_scaled)

pd.DataFrame(y_pred_prob).to_csv("MLP" + suffix + ".csv", index = False)