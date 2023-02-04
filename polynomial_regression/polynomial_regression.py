# Multiple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

# Importing the dataset
dataset = pd.read_csv('../datasets/Summary of Weather.csv', low_memory= False)
X = dataset.iloc[:, [0, 5, 6]].values   # Station, MinTemp, MeanTemp
y = dataset.iloc[:, [4]].values         # MaxTemp

# Preprocessing
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X)
X = scaler.transform(X)

np.set_printoptions(precision=2)
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# 5 degree polynomial features
deg_of_poly = 3
poly = PolynomialFeatures(degree=deg_of_poly)
X_ = poly.fit_transform(X_train)

# Fit linear model
clf = linear_model.LinearRegression()
clf.fit(X_, y_train)

X_2 = poly.fit_transform(X_test)
y_pred = clf.predict(X_2)

# Evaluating the Model Performance
r2 = r2_score(y_test, y_pred)
print("R^2 Score: ", r2)
print("Coefficients: ", clf.coef_)
