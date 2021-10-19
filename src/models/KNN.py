# -*- coding: utf-8 -*-
"""

@author: ngand
"""
import pandas as pd

# Import Dataset
data = pd.read_csv('D:/Senior/Capstone/data-science-enviroment/data/2019/England_2019.csv')
data= data.drop(columns=['Date','Country','Year'])

# Creating Input : All the independent variables
X = data.iloc[:,[0,1,2,4,7]].values
# Creating Output : All the dependent variables
Y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

# #optimization (scaling)
# from sklearn.preprocessing import StandardScaler
# standardScaler_X = StandardScaler()
# X_train = standardScaler_X.fit_transform(X_train)
# X_test = standardScaler_X.transform(X_test)

# Fitting the KNN Classifier
from sklearn.neighbors import KNeighborsClassifier 
knnClassifier = KNeighborsClassifier(n_neighbors = 5)
knnClassifier.fit(X_train, Y_train)

# Predicting the test set results
Y_pred = knnClassifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(Y_test, Y_pred)


# Computing Classification Accuracy
from sklearn.metrics import accuracy_score
classificationAccuracy = accuracy_score(Y_test, Y_pred)
print()

# Main Classification Metrics from the classifier
from sklearn.metrics import classification_report
report = classification_report(Y_test, Y_pred)
print(report)

