import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from MyLinearRegression import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
print(X.shape)
print(np.bincount(y))
# Model 2: Predict petal width using petal length and sepal width

# Split the data into train, validation and test sets with a 80/10/10 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1111, random_state=42, stratify=y_train)

# Create an instance of the linear regression class with a batch size of 32 and a regularization coefficient of 0.01
model_2 = LinearRegression(batch_size=32, regularization=0.01)

# Fit it to the train and validation sets
model_2.fit(X_train, y_train, X_val, y_val)

# Plot the loss history against the step number
plt.plot(model_2.loss_history)

# Label the axes and add a title
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Model 2: Petal width vs Petal length and Sepal width')

# Save the plot as a png file
plt.savefig('model_2.png')

# Print the weights of model 2
print(f"Weights of model 1: {model_2.weights}")

# Train an identical model with a higher regularization coefficient of 0.1
model_2_reg = LinearRegression(batch_size=32, regularization=0.1)

# Fit it to the train and validation sets
model_2_reg.fit(X_train, y_train, X_val, y_val)

# Print the weights of the regularized model
print(f"Weights of regularized model 2: {model_2_reg.weights}")

# Compute and print the difference in weights between the two models
weight_diff = np.abs(model_2.weights - model_2_reg.weights)
print(f"Difference in weights: {weight_diff}")
