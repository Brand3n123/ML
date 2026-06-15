import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3,4]].values

import scipy.cluster.hierarchy as sch #import the libracy to allow for a dendrogram
dendrogram = sch.dendrogram(sch.linkage(x, method='ward')) #linage function take 2 parameters: matrix of featurs and clustering method. method of minimum variance (ward) is preferred

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()