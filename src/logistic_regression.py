import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, norm
from linear_model import LinearModel

# Sigmoid activation function for logistic regression
def sigmoid(z):
    # Numerically stable sigmoid implementation
    if z >= 0:
        return 1 / (1 + np.exp(-z))
    else:
        return np.exp(z) / (1 + np.exp(z))

class LogReg(LinearModel):
    """Logistic Regression model."""

    def fit(self, X_train, y_train):
        """
        Fit the logistic regression model to the training data using Newton-Raphson method.

        Args:
            X_train (array-like): Feature matrix for training.
            y_train (array-like): Target labels for training.
        """
        # Convert inputs to numpy arrays
        X = np.array(X_train)
        y = np.array(y_train)
        # Add intercept term to feature matrix
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # Initialize parameters if not already set
        if self.theta is None:
            self.theta = np.zeros(X.shape[1])

        if self.verbose:
            self.cost = []

        # Newton-Raphson optimization loop
        for iter in range(self.max_iter):
            # Compute hypothesis (predicted probabilities)
            hypothesis = np.array([sigmoid(i @ self.theta) for i in X])
            # Compute gradient
            grad = X.T @ (hypothesis - y)
            # Compute Hessian matrix
            R = (hypothesis * (1 - hypothesis)).reshape(-1, 1)
            hess = X.T @ (X * R)
            # Regularization for invertibility
            hess += 1e-6 * np.eye(X.shape[1])
            old_theta = self.theta.copy()
            # Update parameters using Newton-Raphson step
            self.theta -= inv(hess) @ grad
            # Compute cost (negative log-likelihood)
            if self.verbose:
                self.cost.append(-np.sum(y * np.log(hypothesis) + (1 - y) * np.log(1 - hypothesis)))
            # Check for convergence
            if norm(self.theta - old_theta) <= self.tol:
                if self.verbose:
                    print(f'Convergence reached at iteration {iter+1}')
                break
        self.trained = True
        if self.verbose:
                print("Model trained!")
                plt.plot(self.cost, '--')
                plt.title("Cost vs Iteration")
                plt.show()

    def predict_proba(self, X_test):
        """
        Predict probabilities for the given feature matrix.

        Args:
            X_test (array-like): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        if not self.trained:
            raise Exception('Model is not trained!')

        X = np.asarray(X_test)
        # Add intercept term to feature matrix
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # Return predicted probabilities
        return np.array([sigmoid(i @ self.theta) for i in X])
    
    def predict(self, X_test):
        """
        Predict class labels for the given feature matrix.

        Args:
            X_test (array-like): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted class labels (0 or 1).
        """
        if not self.trained:
            raise Exception('Model is not trained!')

        # Return predicted labels (0 or 1) using 0.5 threshold
        return (self.predict_proba(X_test) >= 0.5).astype(int)
    
    def score(self, X_test, y_test):
        """
        Compute the accuracy of the model on the given test data and labels.

        Args:
            X_test (array-like): Feature matrix for testing.
            y_test (array-like): True labels for testing.

        Returns:
            float: Accuracy (proportion of correct predictions).
        """
        if not self.trained:
            raise Exception('Model is not trained!')
        
        # Return accuracy as the proportion of correct predictions
        return self._score(X_test, y_test, 'accuracy')