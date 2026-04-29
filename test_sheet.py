import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

data_set = pd.read_csv('test_sheet.csv')
x = data_set.iloc[:, :-1].values
y = data_set.iloc[:, -1:].values

imputer_object = SimpleImputer(missing_values= np.nan, strategy= "mean")
imputer_object.fit(x[:, 1:3])
x[:, 1:3] = imputer_object.transform(x[:, 1:3])


'''
#Encoding the Independant variable
'''

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
transformer = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
x = np.array(transformer.fit_transform(x))

'''
#Encoding the Dependent Variable
'''

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print(y)