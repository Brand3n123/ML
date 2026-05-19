import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')
x = dataset.iloc [:, :-1].values
y = dataset.iloc [:, -1].values

from sklearn.preprocessing import StandardScaler
scaled_x = StandardScaler()
x = scaled_x.fit_transform(x)
scaled_y = StandardScaler()
y = scaled_y.fit_transform(y)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x, y)



print(x) 