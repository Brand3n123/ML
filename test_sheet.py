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

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
