# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('../datasets/Summary of Weather.csv', low_memory= False)
x1 = dataset.iloc[:, [5]].values  # MinTemp
x2 = dataset.iloc[:, [6]].values  # MeanTemp
x3 = dataset.iloc[:, [2]].values  # Precip
y = dataset.iloc[:, [4]].values   # MaxTemp
x3[x3 == ['T']] = 0
x3 = x3.astype(np.float64)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, y_train, y_test = train_test_split(x1, x2, x3, y, test_size=1/3, random_state=0)

"""
Var 1 - MinTemp
"""
# Training the Simple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X1_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X1_test)

# Visualising the Training set results
plt.scatter(X1_train, y_train, color='lightcoral')
plt.scatter(X1_test, y_test, color='red')
plt.plot(X1_train, regressor.predict(X1_train), color='blue')
plt.title('MaxTemp vs MinTemp (Test set)')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()

# Evaluating the Model Performance
r2 = r2_score(y_test, y_pred)
print("MinTemp R^2 Score: ", r2)

"""
Var 2 - MeanTemp
"""
# Training the Simple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X2_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X2_test)

# Visualising the Training set results
plt.scatter(X2_train, y_train, color='limegreen')
plt.scatter(X2_test, y_test, color='green')
plt.plot(X2_train, regressor.predict(X2_train), color='blue')
plt.title('MaxTemp vs MeanTemp (Test set)')
plt.xlabel('MeanTemp')
plt.ylabel('MaxTemp')
plt.show()

# Evaluating the Model Performance
r2 = r2_score(y_test, y_pred)
print("MeanTemp R^2 Score: ", r2)

"""
Var 3 - Precip
"""
# Training the Simple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X3_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X3_test)

# Visualising the Training set results
plt.scatter(X3_train, y_train, color='navajowhite')
plt.scatter(X3_test, y_test, color='darkorange')
plt.plot(X3_train, regressor.predict(X3_train), color='blue')
plt.title('MaxTemp vs Precip (Test set)')
plt.xlabel('Precip')
plt.ylabel('MaxTemp')
plt.show()

# Evaluating the Model Performance
r2 = r2_score(y_test, y_pred)
print("Precip R^2 Score: ", r2)
