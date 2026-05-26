import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\weeks\Desktop\GitHub\ML\ML\Machine-Learning-A-Z-Codes-Datasets\Part 3 - Classification\Section 14 - Logistic Regression\Python\Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train) #fit_transform so the feature scaler learns the scaling necessary (fits)
x_test = sc.transform(x_test) #no fit because we don't wan't to also learn from (fit) the test set, only scale on the existing fit

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(p = 2)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
#print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm) #confusion matrix
print(accuracy_score(y_test, y_pred)) #93% accuracy 