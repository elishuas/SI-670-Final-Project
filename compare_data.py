import sys
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# from keras import models
# from keras import layers
# from tensorflow.keras.utils import to_categorical
import pandas as pd
# from tensorflow.keras.metrics import AUC, Recall, Precision 
from sklearn.model_selection import train_test_split


old_ts = pd.read_csv('data/timeseries_readyformodel.csv')

new_ts = pd.read_csv('data/train_ts.csv')

print(old_ts.shape)

print(new_ts.shape)

print("Huh")