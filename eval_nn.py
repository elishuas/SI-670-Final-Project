import torch
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

data_dir = paths['data_dir']
data = pd.read_csv(data_dir + "timeseries_readyformodel.csv")

y = data['died']
X = data.drop(columns = ['died', 'patientunitstayid']).to_numpy().astype('float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=607)


X_train_scaled = X_train
X_test_scaled = X_test

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

X_tst_ts = X_test_scaled.reshape((X.shape[0], 1, X_test_scaled.shape[1]))

model_name = "LSTM_model/"

model = load_model(model_name)

model.summary()

for layer in model.layers:
    print(layer.input_shape)
results = model.evaluate(X_tst_ts, y_test_cat)

print(results)