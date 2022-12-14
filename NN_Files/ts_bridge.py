import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import Utils as U


paths = U.load_paths()


ts = pd.read_csv(paths['data_dir'] + 'timeseries_readyformodel.csv')

train = pd.read_csv(paths['data_dir'] + 'train_new.csv')

test = pd.read_csv(paths['data_dir'] + 'test_new.csv')

left_cols = [c for c in train.columns if c in ts.columns and c != 'patient']


train_ts = pd.merge(ts, train.drop(columns = left_cols), how = 'left', left_on = 'patient', right_on = 'patient')

test_ts = pd.merge(ts, test.drop(columns = left_cols), how = 'left', left_on = 'patient', right_on = 'patient')

test_ts.to_csv(paths['data_dir'] + 'test_ts.csv', index = False)
train_ts.to_csv(paths['data_dir'] + 'train_ts.csv', index = False)