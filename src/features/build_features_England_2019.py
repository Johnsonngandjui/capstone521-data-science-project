# -*- coding: utf-8 -*-
"""
Created on Wed Oct  13 09:48:53 2021

@author: Johnson 
"""

import numpy as np
import pandas as pd


# Importing the dataset
data = pd.read_csv('D:/Senior/Capstone/data-science-enviroment/data/2019/England_2019.csv')
data= data.drop(columns=['Date','Country','Year'])

# Creating Input : All the independent variables
X = data.iloc[:, :-1].values
# Creating Output : All the dependent variables
Y = data.iloc[:,-1].values


#used this to pick what feature to run regresion on
# X = np.array(X[:,[3,4,5,6,7]], dtype = 'float')

# Splitting data into training and test set

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)



# # Creating a Multiple Linear Regression Model to training set
# from sklearn.linear_model import LinearRegression
# regression = LinearRegression()
# regression.fit(X_train, Y_train)


# # Predict the Test results
# Y_pred = regression.predict(X_test)



# """ Building an Optimal Model using Backward Elimination """

X = np.append(arr = np.ones((270,1), dtype = int), values = X,  axis = 1)

# Declare an optimal matrix of features 
X_opt = X[:,:]
#   Back Ward Selection=============================================================================

# # Using statsmodels library to create a model for MLR
# import statsmodels.regression.linear_model as sm
# regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit()

# # print(regression_OLS.summary())
# # #BackWard Selection
# X_opt = X[:, [1, 2, 3, 4, 5, 6, 7, 9,10]]
# regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
# print(regression_OLS.summary())

# X_opt = X[:, [1, 2, 3, 4, 6, 7, 9,10]]
# regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
# print(regression_OLS.summary())

# X_opt = X[:, [1, 2, 3, 4, 6, 7, 10]]
# regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
# print(regression_OLS.summary())

# X_opt = X[:, [1, 2, 3, 4, 10]]
# regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
# print(regression_OLS.summary())

# X_opt = X[:, [1, 2, 3, 10]]
# regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
# print(regression_OLS.summary())
# =============================================================================




# Creating a Simple Linear Regression Model using the training data
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train, Y_train)

y_pred = regression.predict(X_test)
x_pred = regression.predict(X_train)

#comparing actual vs predicted 
df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})


from sklearn.metrics import mean_squared_error, r2_score

#mse
print('MSE',mean_squared_error(Y_train, x_pred))
#RSquared
print('Rsquared',r2_score(Y_train, x_pred))

# Using statsmodels library to create a model for MLR
import statsmodels.regression.linear_model as sm
regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit()

print(regression_OLS.summary())
