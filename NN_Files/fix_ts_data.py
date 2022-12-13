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


X_train_scaled = X_train.to_numpy()
X_test_scaled = X_test.to_numpy()

X_tr_ts = U.to_timeseries(X_train_scaled)
X_tst_ts =  U.to_timeseries(X_test_scaled)

pd.DataFrame(X_tr_ts.reshape(X_train.shape[0], -1)).to_csv("unscaled_train_ts.csv")
pd.DataFrame(X_tst_ts.reshape(X_test.shape[0], -1)).to_csv("unscaled_test_ts.csv")