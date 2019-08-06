from sklearn.preprocessing import MinMaxScaler

# Import libraries
import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Defaults
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams.update({'font.size': 12})
plt.style.use('ggplot')

data2 = pd.read_csv('~//Trane/bad/00C0A6F4.csv')
data2.dropna(inplace=True)
data2.drop('time', axis='columns', inplace=True)
data2.drop('Tsph', axis='columns', inplace=True)
data2.drop('RHi', axis='columns', inplace=True)
data2.drop('RHo', axis='columns', inplace=True)
data2.drop('Hi', axis='columns', inplace=True)
data2.drop('Ho', axis='columns', inplace=True)
data2.drop('DToi', axis='columns', inplace=True)
data2.drop('DAHoi', axis='columns', inplace=True)
# data2.iloc[:, 0] = pd.to_datetime(data2.iloc[:, 0])
# data2.iloc[:, 0] = data2.iloc[:, 0].diff().dt.total_seconds()
data2.dropna(inplace=True)

normalized_data2=(data2-data2.min())/(data2.max()-data2.min())
normalized_data2.fillna(0)

normalized_data2.plot()

plt.show()


scaler = MinMaxScaler()
ecg_np_data = scaler.fit_transform(data2)
ecg_np_data.plot()

plt.show()

# print(ecg_data.head())
# ecg_np_data = ecg_data.as_matrix()
data2 = data2.as_matrix()
scaler = MinMaxScaler()
# ecg_np_data = scaler.fit_transform(ecg_np_data)
ecg_np_data = scaler.fit_transform(data2)
print(ecg_np_data.shape)

