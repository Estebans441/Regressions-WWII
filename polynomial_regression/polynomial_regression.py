# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('../datasets/Summary of Weather.csv', low_memory= False)
x1 = dataset.iloc[:, 5:6].values  # MinTemp
x2 = dataset.iloc[:, 6:7].values  # MeanTemp
x3 = dataset.iloc[:, 2:3].values  # Precip
y = dataset.iloc[:, 4:5].values   # MaxTemp
x3[x3 == ['T']] = 0
x3 = x3.astype(np.float64)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(np.column_stack((x1,x2,x3)), y, test_size=0.33, random_state=0)

# Fitting Polynomial Regression to the dataset
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(x_train)
poly_reg.fit(X_poly, y_train)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_train)

# Predicting the Test set results
X_test_poly = poly_reg.transform(x_test)
y_pred = lin_reg.predict(X_test_poly)

# Evaluating the Model Performance
from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("Root Mean Squared Error: ", rmse)
print("R^2 Score: ", r2)
