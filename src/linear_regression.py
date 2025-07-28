import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.linalg import inv, norm

from linear_model import LinearModel

class LinReg(LinearModel):
    """Linear Regression model."""

    def fit(self, X_train, y_train, method='Analytical'):
        """
        Fit the linear regression model to the training data using the specified method.

        Args:
            X_train (array-like): Feature matrix for training.
            y_train (array-like): Target values for training.
            method (str): Training method ('Analytical', 'GD', or 'SGD').
        """
        X = np.asarray(X_train)
        y = np.asarray(y_train)
        # Add intercept term to feature matrix
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        method = method.upper()
        if method == 'ANALYTICAL':
            # Analytical solution using normal equation
            reg = 1e-6 * np.eye(X.shape[1]) # Regularization term to avoid singular matrix
            self.theta = inv((X.T @ X) + reg) @ X.T @ y
            self.trained = True
            if self.verbose:
                print("Model trained!")
        elif method == 'GD':
            # Gradient Descent
            if self.theta is None:
                self.theta = np.zeros(X.shape[1])
            if self.verbose:
                self.cost = []
            for iter in range(self.max_iter):
                old_theta = self.theta.copy()
                # Compute gradient
                grad = X.T @ (X @ self.theta - y)
                # Update parameters
                self.theta -= self.lr * grad
                if self.verbose:
                    # Compute and save cost for plotting
                    cost = 0.5 * np.sum((X @ self.theta - y) ** 2)
                    self.cost.append(cost)
                # Check for convergence
                if norm(self.theta - old_theta) <= self.tol:
                    break
            self.trained = True
            if self.verbose:
                print("Model trained!")
                plt.plot(self.cost, '--')
                plt.title("Cost vs Iteration")
                plt.show()
        elif method == 'SGD':
            # Stochastic Gradient Descent
            if self.theta is None:
                self.theta = np.zeros(X.shape[1])
            if self.verbose:
                self.cost = []
            for iter in range(self.max_iter):
                # Randomly select a data point
                idx = random.randint(0, X.shape[0] - 1)
                old_theta = self.theta.copy()
                # Compute gradient for single sample
                grad = X[idx] * (X[idx] @ self.theta - y[idx])
                # Update parameters
                self.theta -= self.lr * grad
                if self.verbose:
                    # Compute and save cost for plotting
                    cost = 0.5 * np.sum((X @ self.theta - y) ** 2)
                    self.cost.append(cost)
                # Check for convergence
                if norm(self.theta - old_theta) <= self.tol:
                    break
            self.trained = True
            if self.verbose:
                print("Model trained!")
                plt.plot(self.cost, '--')
                plt.title("Cost vs Iteration")
                plt.show()
        else:
            raise ValueError('Method must be "Analytical", "GD", or "SGD".')

    def predict(self, X_test):
        """
        Predict target values for the given feature matrix.

        Args:
            X_test (array-like): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted target values.
        """
        if not self.trained:
            raise Exception('Model is not trained.')
        X = np.asarray(X_test)
        # Add intercept term to feature matrix
        X = np.hstack((np.ones((X_test.shape[0], 1)), X))
        # Return predictions
        return X @ self.theta
    
    def score(self, X_test, y_test):
        """
        Compute the model score using MSE.

        Args:
            X_test (array_like): Feature matrix for testing.
            y_test (array like): Target values for testing.

        Returns:
            float: MSE of the test
        """
        if not self.trained:
            raise Exception('Model is not trained!')

        return self._score(X_test, y_test, metric='mse')
