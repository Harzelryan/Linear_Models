import numpy as np
class LinearRegression:
    def __init__(self, max_epochs=1000, batch_size=32, steps=None,
                 regularization=0.0, learning_rate=0.01,
                 decay_factor=0.9, patience=3):
        # Initialize the parameters
        self.max_epochs = max_epochs # The maximum number of epochs to train
        self.batch_size = batch_size # The size of each batch
        self.steps = steps # The number of steps to train
        self.regularization = regularization # The regularization coefficient
        self.learning_rate = learning_rate # The initial learning rate
        self.decay_factor = decay_factor # The factor to reduce the learning rate by every epoch
        self.patience = patience # The number of epochs without improvement to stop training

        # Initialize some variables for early stopping and learning rate decay
        self.best_loss = np.inf # The best validation loss so far
        self.best_weights = None # The best weights so far
        self.best_bias = None # The best bias so far
        self.no_improvement = 0 # The number of epochs without improvement in validation loss

        # Initialize some variables to store the loss history and the current step
        self.loss_history = [] # A list of losses averaged over each batch size for each step
        self.current_step = 0 # The current step number

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit a linear model.
        Parameters:
        -----------
        X_train: numpy.ndarray
            The input data for training.
        y_train: numpy.ndarray
            The target data for training.
        X_val: numpy.ndarray or None
            The input data for validation.
            If None, no validation is performed.
        y_val: numpy.ndarray or None
            The target data for validation.
            If None, no validation is performed.
        """
        # Initialize the weights and bias randomly based on the shape of X_train and y_train.
        n_features = X_train.shape[1]
        n_samples = X_train.shape[0]
        self.weights = np.random.randn(n_features)
        self.bias = np.random.randn()

        # Loop over the epochs until the maximum number is reached or early stopping is triggered.
        for epoch in range(self.max_epochs):
            # Shuffle the training data and split it into batches.
            permutation = np.random.permutation(n_samples)
            X_train = X_train[permutation]
            y_train = y_train[permutation]
            batches = [(X_train[i:i+self.batch_size], y_train[i:i+self.batch_size]) for i in range(0, n_samples, self.batch_size)]

            # Loop over the batches and update the weights and bias using gradient descent.
            for X_batch, y_batch in batches:
                # Compute the predictions and the errors for the current batch.
                y_pred = self.predict(X_batch)
                errors = y_pred - y_batch

                # Compute the gradients of the loss function with respect to the weights and bias.
                grad_w = (2 / self.batch_size) * (X_batch.T @ errors) + (2 * self.regularization * self.weights)
                grad_b = (2 / self.batch_size) * np.sum(errors)

                # Update the weights and bias using the learning rate and the gradients.
                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

                # Compute and record the loss averaged over the batch size for the current step.
                batch_loss = self.score(X_batch, y_batch)
                self.loss_history.append(batch_loss)

                # Increment the current step number.
                self.current_step += 1

                # If the number of steps is specified, check if it is reached and stop training if so.
                if self.steps is not None and self.current_step >= self.steps:
                    print(f"Reached {self.steps} steps at epoch {epoch}")
                    return

            # If validation data is provided, evaluate the loss on the validation set after each epoch.
            if X_val is not None and y_val is not None:
                val_loss = self.score(X_val, y_val)

                # Check if the validation loss has improved or not.
                if val_loss < self.best_loss:
                    # Save the current weights and bias as the best ones so far.
                    self.best_loss = val_loss
                    self.best_weights = self.weights.copy()
                    self.best_bias = self.bias.copy()

                    # Reset the counter for epochs without improvement.
                    self.no_improvement = 0
                else:
                    # Increment the counter for epochs without improvement.
                    self.no_improvement += 1

                    # If the counter reaches the patience limit, stop training and restore the best weights and bias.
                    if self.no_improvement == self.patience:
                        print(f"Early stopping at epoch {epoch}")
                        # Restore the best weights and bias
                        self.weights = self.best_weights
                        self.bias = self.best_bias
                        return

            # Reduce the learning rate by the decay factor after each epoch.
            self.learning_rate *= self.decay_factor

        # Return the final weights, bias and loss history
        return self.weights, self.bias, self.loss_history

    def predict(self, X):
        """Predict using the linear model.
        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        # Compute the linear combination of the input features and the weights, plus the bias.
        return X @ self.weights + self.bias

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.
        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        # Compute the predictions and the errors for the given data.
        y_pred = self.predict(X)
        errors = y_pred - y

        # Compute and return the mean squared error without the regularization term.
        return (1 / X.shape[0]) * (errors.T @ errors)
