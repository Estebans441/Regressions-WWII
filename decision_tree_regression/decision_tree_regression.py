# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('../datasets/Summary of Weather.csv', low_memory= False)
x1 = dataset.iloc[:, [5]].values  # MinTemp
x2 = dataset.iloc[:, [6]].values  # MeanTemp
x3 = dataset.iloc[:, [2]].values  # Precip
y = dataset.iloc[:, [4]].values   # MaxTemp
x3[x3 == ['T']] = 0
x3 = x3.astype(np.float64)

"""
Var 1 - MinTemp
"""
# Training the Simple Linear Regression model on the Training set
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x1, y)

# Visualising the Training set results
X_grid = np.arange(min(x1), max(x1), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x1, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('MaxTemp vs MinTemp (Decision Tree Regression)')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()

"""
Var 2 - MeanTemp
"""
# Training the Simple Linear Regression model on the Training set
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x2, y)

# Visualising the Training set results

X_grid = np.arange(min(x2), max(x2), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x2, y, color='green')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('MaxTemp vs MeanTemp (Decision Tree Regression)')
plt.xlabel('MeanTemp')
plt.ylabel('MaxTemp')
plt.show()

"""
Var 3 - Precip
"""
# Training the Simple Linear Regression model on the Training set
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x3, y)

# Visualising the Training set results
X_grid = np.arange(min(x3), max(x3), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x3, y, color='darkorange')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('MaxTemp vs Precip (Decision Tree Regression)')
plt.xlabel('Precip')
plt.ylabel('MaxTemp')
plt.show()
