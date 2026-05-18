#no feature scaling in SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y), 1) #makes it vertical instead of horizontal

from sklearn.preprocessing import StandardScaler #make a list of all the classes imported in all sheets and why/when
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# what was the point
# of the SVR from a
# ML perspective?

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') # read about kernels (relevance/purpose/differences)
regressor.fit(x, y)


print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1, 1)))
#why did we need to reshape this, and why (-1,1) ?

plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x).reshape(-1, 1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()