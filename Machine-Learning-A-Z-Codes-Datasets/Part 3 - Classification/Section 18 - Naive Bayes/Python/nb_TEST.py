import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train) #fit_transform so the feature scaler learns the scaling necessary (fits)
x_test = sc.transform(x_test) #no fit because we don't wan't to also learn from (fit) the test set, only scale on the existing fit


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train) #trains the model on the inputs/outputs of the training data


#testing for accuracy here in syntax/personal understanding
#print(classifier.predict([x_test[0, :]])) #predict the ourputs of the test set's first row (30, 87000)
#print(classifier.predict(sc.transform([[30,87000]]))) #apply the feature scaling object on the specific prediction to ensure accuracy in syntax thus far

y_pred = classifier.predict(x_test)

#For visualizing the predictions vs the actuals
#reshaped_prediction = y_pred.reshape(len(y_pred), 1) #Vertical prediction
#reshaped_actual = y_test.reshape(len(y_test), 1) #Vertical actual
#concatenated_results = np.concatenate((reshaped_prediction, reshaped_actual), 1) #Combined for comparison
#print(concatenated_results)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm) #89/100 with 11 wrong predictions
print(accuracy_score(y_test, y_pred)) #89% accuracy 