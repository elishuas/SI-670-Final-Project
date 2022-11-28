import torch
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


from keras import models
from keras import layers

from tensorflow.keras.utils import to_categorical
import pandas as pd

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

y_train = train['died']
y_test = test['died']

X_train = train.drop(columns = 'died')
X_test = test.drop(columns = 'died')

# scaler = StandardScaler()
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

net = models.Sequential()
net.add(layers.Dense(16, activation = 'tanh', input_shape = X_train_scaled[0].shape))
net.add(layers.Dense(32, activation = 'tanh'))
net.add(layers.Dense(64, activation = 'tanh')) # Up to 82%
net.add(layers.Dense(128, activation = 'tanh')) # Up to 82.5%
net.add(layers.Dense(256, activation = 'tanh')) # 
net.add(layers.Dense(512, activation = 'tanh')) # 
# net.add(layers.Conv1D(256, kernel_size = 2))# 
net.add(layers.Dense(256, activation = 'relu')) # 85% for flat 256, 3 tanh layers; down to 84% when increasing powers of 2
# net.add(layers.Dense(32, activation = 'relu'))
# net.add(layers.Dense(16, activation = 'relu'))
# net.add(layers.Dense(8, activation = 'relu'))

# Output Layer
net.add(layers.Dense(2, activation = 'sigmoid'))

net.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['Accuracy'])

net.fit(X_train_scaled, y_train_cat, epochs = 1000, batch_size = 200)#, verbose = False)

test_accuracy = net.evaluate(X_test_scaled, y_test_cat)[1]

print(f'Test Accuracy: {test_accuracy}')
