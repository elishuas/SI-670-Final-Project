import pandas as pd


path = "/Users/stephentoner/Desktop/eicu-research-database/"

ts_lab = pd.read_csv(path + 'timeserieslab.csv')
ts_resp =  pd.read_csv(path + 'timeseriesresp.csv')
ts_periodic =  pd.read_csv(path + 'timeseriesperiodic.csv')
ts_aperiodic =  pd.read_csv(path + 'timeseriesaperiodic.csv')

# merged = pd.concat((ts_lab, ts_resp, ts_))