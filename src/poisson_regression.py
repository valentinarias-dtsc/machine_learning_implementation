import numpy as np
import matplotlib.pyplot as plt
from linear_model import LinearModel

class PoissonRegression(LinearModel):
    """Poisson Regression Model."""

    def fit(self, X_train, y_train):
        """
        Fit the Poisson regression model to the training data

        Args:
            X_train (array-like): Feature matrix for training.
            y_train (array-like): Target values for training.
        """
        X = np.array(X)
        y = np.array(y)

        # Add intercept term to feature matrix
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        # Initialize theta with small random values
        if self.theta is None:
            np.random.seed(0)
            theta = np.random.normal(scale=1e-4, size=X.shape[1])
        else:
            theta = self.theta.copy()

        if self.verbose:
            self.history_ll = []

        prev_ll = np.inf

        # Gradient Ascent loop
        for i in range(self.max_iter):

            grad = self._gradient(X, y, theta)
            theta += self.lr * grad
            ll = self._log_likelihood(X, y, theta)

            if self.verbose:
                self.history_ll.append(ll)

            if np.abs(prev_ll - ll) < self.tol:
                if self.verbose:
                    print(f'Convergence reached at iteration {i}')
                break

            prev_ll = ll
        
        # Save parameters 
        self.theta = theta
        self.traioned = True
        if self.verbose:
            print('Model trained!')
            plt.plot(self.history_ll, label='log Likelihood', color='black', linewidth=2)
            plt.title('log Likelihood vs Iterations')
            plt.xlabel('Iterations')
            plt.ylabel('log Likelihood')
            plt.grid()
    
    def predict(self, X):
        """
        Predict target values for the given feature matrix.

        Args:
            X_test (array-like): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        X = np.array(X)
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        if not self.trained:
            raise Exception('Model is not trained.')
        
        # Hypothesis for Poisson distributions
        eta = X @ self.theta
        return np.exp(eta)

    def score(self, X_test, y_test):
        """
        Compute the model score using R2.

        Args:
            X_test (array_like): Feature matrix for testing.
            y_test (array like): Target values for testing.

        Returns:
            float: R2 of the test
        """
        if not self.trained:
            raise Exception('Model is not trained.')

        return self._score(X_test, y_test, metric='r2')

    def _log_likelihood(self, X, y, theta):
        # Local method to compute the log likelihood
        sub_eta = np.clip(X @ theta, -500, 500) # Prevent overflow
        return np.sum(y * sub_eta - np.exp(sub_eta))
    
    def _gradient(self, X, y, theta):
        # Local method to compute the gradient of the log likelihood
        sub_eta = np.clip(X @ theta, -500, 500) # Prevent overflow
        return X.T @ (y - np.exp(sub_eta))