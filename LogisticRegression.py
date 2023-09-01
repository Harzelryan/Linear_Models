import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions


class LogisticRegression:
    def __init__(self):
        self.weights = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, learning_rate=0.01, num_iterations=1000):
        X = np.insert(X, 0, 1, axis=1)  # Add a column of 1s for the bias term
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)

        for _ in range(num_iterations):
            scores = np.dot(X, self.weights)
            predictions = self._sigmoid(scores)
            gradient = np.dot(X.T, y - predictions)
            self.weights += (learning_rate / num_samples) * gradient

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)  # Add a column of 1s for the bias term
        scores = np.dot(X, self.weights)
        probabilities = self._sigmoid(scores)
        return np.round(probabilities).astype(int)


# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an instance of LogisticRegression and fit the training data
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Predict the test data
y_pred = lr.predict(X_test)

# Compare the predicted labels with the ground truth labels
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

# Visualize the decision regions for petal length/width features
petal_indices = [2, 3]
X_petal = X[:, petal_indices]
X_petal_train = X_train[:, petal_indices]
X_petal_test = X_test[:, petal_indices]

# Fit and predict using petal length/width features
lr_petal = LogisticRegression()
lr_petal.fit(X_petal_train, y_train)
y_pred_petal = lr_petal.predict(X_petal_test)

# Plot decision regions for petal length/width features
plot_decision_regions(X_petal_test, y_test, clf=lr_petal)
plt.title('Logistic Regression - Petal Length/Width Features')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.show()

# Visualize the decision regions for sepal length/width features
sepal_indices = [0, 1]
X_sepal = X[:, sepal_indices]
X_sepal_train = X_train[:, sepal_indices]
X_sepal_test = X_test[:, sepal_indices]

# Fit and predict using sepal length/width features
lr_sepal = LogisticRegression()
lr_sepal.fit(X_sepal_train, y_train)
y_pred_sepal = lr_sepal.predict(X_sepal_test)

# Plot decision regions for sepal length/width features
plot_decision_regions(X_sepal_test, y_test, clf=lr_sepal)
plt.title('Logistic Regression - Sepal Length/Width Features')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()
