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

train = pd.read_csv(data_dir + "train.csv")
test = pd.read_csv(data_dir + "test.csv")
data = pd.concat([train, test])
scaler = StandardScaler()
# scaler = MinMaxScaler()


y_train = np.asarray(train['died']).astype('float32').reshape((-1,1))
X_train = train.drop(columns = ['died']).to_numpy().astype('float32')


y_test = np.asarray(test['died']).astype('float32').reshape((-1,1))
X_test = test.drop(columns = ['died']).to_numpy().astype('float32')


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = load_model(paths['models_path'] + "MLP_modelbest_long")

model.summary()

for layer in model.layers:
    print(layer.input_shape)
results = model.evaluate(X_test_scaled, y_test)

print(results)


train_pred_probs = model.predict(X_train_scaled)
test_pred_probs = model.predict(X_test_scaled)

train_results = U.compute_metrics(train_pred_probs, y_train)
test_results = U.compute_metrics(test_pred_probs, y_test)

train_precision, train_recall, _ = (
    U.precision_recall_curve(y_train, 
                           model.predict(X_train_scaled))
)
test_precision, test_recall, _ = (
    U.precision_recall_curve(y_test, 
                           model.predict(X_test_scaled))
)

train_auprc = U.auc(train_recall, train_precision)
test_auprc = U.auc(test_recall, test_precision)

print(train_results)
print(test_results)
print(train_auprc)
print(test_auprc)

U.save_metrics({"train": train_results, "test": test_results},
                [train_auprc, test_auprc],
                model_name ="MLP",
                save = True)
