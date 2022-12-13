import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import Utils as U


# Load aggregated eICU patient table

paths = U.load_paths()

data_dir = paths['data_dir']
flat = pd.read_csv(data_dir + 'preprocessed_flat.csv')

time_series = pd.read_csv(data_dir + 'preprocessed_timeseries.csv')

patient = pd.read_csv('raw_agg_patient_table.csv')

flat.head()

ts_data = pd.merge(time_series, patient, how = 'left', left_on = 'patient', right_on = 'patientunitstayid').dropna()

ts_data.to_csv(data_dir + "new_timeseries_readyformodel.csv", index= False)

patient['patientunitstayid'] = patient['patientunitstayid'].astype(int)

data = pd.merge(time_series, flat.drop(columns = 'age'), how = 'left', left_on = 'patient', right_on = 'patient')

dup_cols = [c for c in patient.columns if c in data.columns]

data = pd.merge(data, patient.drop(columns = dup_cols), how = 'left', left_on = 'patient', right_on = 'patientunitstayid')

cols_to_recode = ['urine', 'wbc', 'temperature', 'respiratoryrate', 'sodium',
                  'heartrate', 'meanbp', 'ph', 'creatinine', 'albumin', 
                  'glucose', 'bilirubin', 'fio2', 'pao2', 'pco2', 'bun', 
                  'meanapachescore', 'meanpredictedicumortality', 
                  'meanpredictediculos']

for col in cols_to_recode: 
    data.loc[data[col] == -1, col] = np.nan

def get_categorical_percentages(df):
    """
    This function computes the percentages of categories for each categorical
    variable in the data.
    
    :param df: data
    :type df: pandas dataframe
    
    :return: prints each variable with their categories and respective 
    proportions
    """
    cat_df = df.select_dtypes(include='category')
    for var in cat_df.columns:
        perc = df[var].value_counts(normalize=True)
        print(var)
        print(perc)


data.age = data.age.astype('category')

# Recode age, teaching status and gender variables
data.age = data.age.cat.rename_categories({'> 89': 90})
data.age = pd.to_numeric(data.age, errors='coerce')

data['age0-9'] = np.nan
data.loc[(data.age >= 0) & (data.age <= 9), 'age0-9'] = 1
data.loc[(data.age > 9), 'age0-9'] = 0
data['age0-9'] = data['age0-9'].astype('category')

data['age10-19'] = np.nan
data.loc[(data.age >= 10) & (data.age <= 19), 'age10-19'] = 1
data.loc[(data.age > 19), 'age10-19'] = 0
data['age10-19'] = data['age10-19'].astype('category')

data['age20-29'] = np.nan
data.loc[(data.age >= 20) & (data.age <= 29), 'age20-29'] = 1
data.loc[(data.age > 29), 'age20-29'] = 0
data['age20-29'] = data['age20-29'].astype('category')

data['age30-39'] = np.nan
data.loc[(data.age >= 30) & (data.age <= 39), 'age30-39'] = 1
data.loc[(data.age > 39), 'age30-39'] = 0
data['age30-39'] = data['age30-39'].astype('category')

data['age40-49'] = np.nan
data.loc[(data.age >= 40) & (data.age <= 49), 'age40-49'] = 1
data.loc[(data.age > 49), 'age40-49'] = 0
data['age40-49'] = data['age40-49'].astype('category')

data['age50-59'] = np.nan
data.loc[(data.age >= 50) & (data.age <= 59), 'age50-59'] = 1
data.loc[(data.age > 59), 'age50-59'] = 0
data['age50-59'] = data['age50-59'].astype('category')

data['age60-69'] = np.nan
data.loc[(data.age >= 60) & (data.age <= 69), 'age60-69'] = 1
data.loc[(data.age > 69), 'age60-69'] = 0
data['age60-69'] = data['age60-69'].astype('category')

data['age70-79'] = np.nan
data.loc[(data.age >= 70) & (data.age <= 79), 'age70-79'] = 1
data.loc[(data.age > 79), 'age70-79'] = 0
data['age70-79'] = data['age70-79'].astype('category')

data['age80-89'] = np.nan
data.loc[(data.age >= 80) & (data.age <= 89), 'age80-89'] = 1
data.loc[(data.age > 89), 'age80-89'] = 0
data['age80-89'] = data['age80-89'].astype('category')

data['age>89'] = np.nan
data.loc[(data.age > 89), 'age>89'] = 1
data.loc[(data.age <= 89), 'age>89'] = 0
data['age>89'] = data['age>89'].astype('category')
data.drop(columns='age', inplace=True)

data.teachingstatus = np.where(data.teachingstatus == 't', 1, 0)
data.teachingstatus = data.teachingstatus.astype('category')

data['gendermale'] = np.nan
data.loc[data.gender == 'Male', 'gendermale'] = 1
data.loc[data.gender == 'Female', 'gendermale'] = 0
data.gendermale = data.gendermale.astype('category')
data.drop(columns='gender', inplace=True)

data.drop(columns=['aids', 'lymphoma', 'unitstaytype'], inplace=True)


# One hot encode categorical variables
categorical_df = data.select_dtypes(include='category')
cols_to_encode = [col for col in categorical_df.columns 
                  if categorical_df[col].nunique() > 2]
ind_vars = pd.get_dummies(categorical_df.loc[:, cols_to_encode])
ind_vars = ind_vars.apply(lambda x: x.astype('category'))

# Remove hospital region from list of columns to drop because it will be used 
# for stratified sampling later
cols_to_encode.remove('hospitalregion')

data = data.drop(columns=cols_to_encode)
data = pd.concat([data, ind_vars], axis=1)