# Importing the necessary libraries
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

# Load the dataset
dataset = pd.read_csv('test_sheet.csv')

# Identify missing data (assumes that missing data is represented as NaN)
missing_data = dataset.isnull().sum().sum() #isnull() converts every value to True/False if Null/Not, sum() shows total Null values by column, second sum() shows total null values
#did not need numPy array

# Identify missing data (assumes that missing data is represented as NaN)
print(missing_data)

# Configure an instance of the SimpleImputer class
imputer_class = SimpleImputer(missing_values = np.nan, strategy= "mean")

# Fit the imputer on the DataFrame
imputer_class.fit(dataset[:]) #fits imputer to every row/column - entire dataset *MUST BE NUMERICAL DATA ONLY*

# Apply the transform to the DataFrame
#x = imputer_class.transform(dataset[:]) #applies the transformatin and assigns to variable

#Print your updated matrix of features
print(imputer_class.transform(dataset[:]))