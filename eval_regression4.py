# Script for model 4: Predict petal length using sepal length and sepal width
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
X = df[['sepal length (cm)', 'sepal width (cm)']].values
y = df['petal length (cm)'].values.reshape(-1, 1)

# Split the data into train, validation and test sets with a 80/10/10 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=df['target'])
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1111, random_state=42)


# Create an instance of the linear regression class with the weights and bias from model 4
model_4 = LinearRegression(batch_size=32, regularization=0.01)
model_4.weights=np.array([1.7757, -1.5345])
model_4.bias=np.array([-3.5396])
# Evaluate the model on the test set and print the mean squared error
test_error = model_4.score(X_test, y_test)
print(f"Test error for model 4: {test_error.mean()}")
