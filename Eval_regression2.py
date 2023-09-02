# Script for model 2: Predict petal width using petal length and sepal width
# Import the libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from MyLinearRegression import LinearRegression

# Load the iris dataset and convert it to a dataframe
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Split the data into X and y
X = df[['petal length (cm)', 'sepal width (cm)']].values
y = df['petal width (cm)'].values.reshape(-1, 1)

# Split the data into train, validation and test sets with a 80/10/10 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=df['target'])
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1111, random_state=42)#, stratify = y_train)

# Define a class for the linear regression model that has a predict and score method

# Create an instance of the linear regression class with the weights and bias from model 2
model_2 = LinearRegression(batch_size=32, regularization=0.01)
model_2.weights=np.array([0.4164, -0.3665]) 
model_2.bias=np.array([-0.2403])
# Evaluate the model on the test set and print the mean squared error
test_error = model_2.score(X_test, y_test)
print(f"Test error for model 2: {test_error.mean()}")
