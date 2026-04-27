'''import numpy as np
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

print(y)'''


# Importing the necessary libraries
import pandas as pd 
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# Load the dataset
dataset = pd.read_csv('titanic.csv')

# Identify the categorical data
categorical_features = ["Sex", "Embarked", "Pclass"]
X_columns = ['Sex', 'Embarked', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
X = dataset[X_columns]

# Implement an instance of the ColumnTransformer class
transformer = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), categorical_features)], remainder = "passthrough")

# Apply the fit_transform method on the instance of ColumnTransformer
X = transformer.fit_transform(X)

# Convert the output into a NumPy array
X = np.array(X)

# Use LabelEncoder to encode binary categorical data
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(dataset['Survived'])

# Print the updated matrix of features and the dependent variable vector
print(X)
print(y)