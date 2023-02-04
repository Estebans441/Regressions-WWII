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

# Preprocessing de data of 3rd variable deleting rows equal to T(unknown data)
indexes = np.where(x3[:] == ['T'])
x3 = np.delete(x3, indexes, axis=0).astype(np.float64)
y2 = np.delete(y, indexes, axis=0)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=1/3, random_state=0)
X3_train, X3_test, y2_train, y2_test = train_test_split(x3, y2, test_size=1/3, random_state=0)

"""
Var 1 - MinTemp
"""
# Training the Simple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X1_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X1_test)

# Evaluating the Model Performance
r2 = r2_score(y_test, y_pred)
print("MinTemp")
print("   R^2 Score: ", r2)
print("   Coefficients: ", regressor.coef_)

# Visualising the Training set results
plt.scatter(X1_train, y_train, color='lightcoral')
plt.scatter(X1_test, y_test, color='red')
plt.plot(X1_train, regressor.predict(X1_train), color='blue')
plt.title('MaxTemp vs MinTemp (Test set)')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()

"""
Var 2 - MeanTemp
"""
# Training the Simple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X2_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X2_test)

# Evaluating the Model Performance
r2 = r2_score(y_test, y_pred)
print("MeanTemp")
print("   R^2 Score: ", r2)
print("   Coefficients: ", regressor.coef_)

# Visualising the Training set results
plt.scatter(X2_train, y_train, color='limegreen')
plt.scatter(X2_test, y_test, color='green')
plt.plot(X2_train, regressor.predict(X2_train), color='blue')
plt.title('MaxTemp vs MeanTemp (Test set)')
plt.xlabel('MeanTemp')
plt.ylabel('MaxTemp')
plt.show()

"""
Var 3 - Precip
"""
# Training the Simple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X3_train, y2_train)

# Predicting the Test set results
y_pred = regressor.predict(X3_test)

# Evaluating the Model Performance
r2 = r2_score(y2_test, y_pred)
print("Precip")
print("   R^2 Score: ", r2)
print("   Coefficients: ", regressor.coef_)

# Visualising the Training set results
plt.scatter(X3_train, y2_train, color='navajowhite')
plt.scatter(X3_test, y2_test, color='darkorange')
plt.plot(X3_train, regressor.predict(X3_train), color='blue')
plt.title('MaxTemp vs Precip (Test set)')
plt.xlabel('Precip (mm)')
plt.ylabel('MaxTemp')
plt.show()
