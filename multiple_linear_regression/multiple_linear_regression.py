# Multiple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('../datasets/Summary of Weather.csv', low_memory= False)
X = dataset.iloc[:, [0, 5, 6]].values   # Station, MinTemp, MeanTemp
y = dataset.iloc[:, [4]].values         # MaxTemp
print(X[:5])

"""# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X)
scaled_X = scaler.transform(X)
print(scaled_X[:5])"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Evaluating the Model Performance
r2 = r2_score(y_test, y_pred)
print("R^2 Score: ", r2)
