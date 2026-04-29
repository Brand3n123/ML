import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

#Create the seperate arrays for dependent and independent variables
data_set = pd.read_csv('test_sheet.csv')
x = data_set.iloc[:, :-1].values
y = data_set.iloc[:, -1:].values



#Check first array for missing values and fit/transfor on all rows, columns 1:3 (aka 1 and 2) 
#with speciffic instructions to replace missing "nan" values with the average of all other values
imputer_object = SimpleImputer(missing_values= np.nan , strategy= "mean")
imputer_object.fit(x[:, 1:3])
x[:, 1:3] = imputer_object.transform(x[:, 1:3])


'''
#Encoding the Independant variable
'''
#Encode the IV with One Hot encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
transformer = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
x = np.array(transformer.fit_transform(x))



''' 
#Encoding the Dependent Variable
'''

#encode the DV with Label Encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
#print("\nEncoded IV")
#print(f"{x}\n")

#print("Encoded DV")
#print(f"{y}\n")
"""

'''
#Creating the Training/Test sets
'''
"""
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
"""print(f"x_train - Training Set:\n{x_train}")
print(f"x_test - Test Set:\n{x_test}")

print(f"y_train - Training Set:\n{y_train}")
print(f"y_test - Test Set:\n{y_test}")"""

"""
"""
'''
#Feature Scaling
'''

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print(f"\nTraining Set - Feature Scaled:\n{x_train}")
