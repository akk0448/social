# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 00:45:55 2020

@author: Aniket
"""


#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import Dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 13].values

#Splitting the dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Trying out attribute combinations
dataset["TAXRM"] = dataset["TAX"]/dataset["RM"]

#Finding correlations
corr_matrix = dataset.corr()
corr_medv = corr_matrix['MEDV'].sort_values(ascending = False)
from pandas.plotting import scatter_matrix
attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(dataset[attributes], figsize = (12,8))

dataset.plot(kind = "scatter", x="TAXRM", y="MEDV", alpha=0.8)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)

#Predicting the test dataset
y_pred = regressor.predict(X_test)

from joblib import dump, load
dump(regressor, 'housing.joblib')