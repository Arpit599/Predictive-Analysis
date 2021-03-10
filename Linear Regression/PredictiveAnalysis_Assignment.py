# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 19:32:52 2021

@author: dell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("vehicle_data.csv")
columns_rearranged = ["Car_Name", "Year", "Present_Price", "Kms_Driven", "Fuel_Type", "Seller_Type", "Transmission", "Owner", "Selling_Price"]
dataset = dataset.reindex(columns = columns_rearranged)

X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, 8].values

from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
# oneHotEncoder = OneHotEncoder(categories=[3])
# #X = oneHotEncoder.fit_transform(X).toarray()
X[:,3] = labelEncoder_X.fit_transform(X[:,3])
X[:,4] = labelEncoder_X.fit_transform(X[:,4])
X[:,5] = labelEncoder_X.fit_transform(X[:,5])


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, Y_train)

y_pred = regression.predict(X_test)

import statsmodels.api as sm
X = np.append(arr=np.ones((301, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5,6]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(Y, X_opt).fit()
print(regressor_OLS.summary())

plt.scatter(y_pred, Y_test, color = "red")
plt.plot(Y_train, X_train[:, 0], color = "green")
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()