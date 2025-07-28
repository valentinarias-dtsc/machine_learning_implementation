import numpy as np
from numpy.linalg import inv

from generative_model import GenerativeModel

class LDA(GenerativeModel):
    """Linear Discriminant Analysis Model."""

    def fit(self, X_train, y_train):
        """ 
        Fit the LDA model to the training data.

        Args:
            X_train (array-like): Feature matrix for training.
            y_train (array-like): Target labels for training.
        """

        X = np.asarray(X_train)
        y = np.asarray(y_train)

        # Get unique class labels
        self.K = np.unique(y)
        n, m = X.shape

        # Compute class priors
        self.priors = np.asarray([np.mean(y == k) for k in self.K])

        # Compute class means
        self.mu = np.asarray([np.mean(X[y == k], axis=0) for k in self.K])

        # Initialize shared covariance matrix
        self.sigma = np.zeros((m, m))

        # Compute shared covariance matrix
        for i in range(n):
            diff = (X[i] - self.mu[y[i]]).reshape(-1, 1)
            self.sigma += (diff @ diff.T) / n

        # Add small value to diagonal for numerical stability
        self.sigma += 1e-6 * np.eye(self.sigma.shape[0])

        self.trained = True

    def predict_proba(self, X_test):
        """ 
        Predict the class probabilities for the given feature matrix.

        Args:
            X_test (array-like): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        X = np.asarray(X_test)

        probs = []
        for x in X:
            probs_k = []
            for i in range(len(self.K)):
                # Compute the posterior (unnormalized) for each class
                x_mu = X - self.mu[i]
                post = np.exp(-0.5 * x_mu @ inv(self.sigma) @ x_mu.T)
                probs_k.append(post * self.priors[i])
            probs_k = np.asarray(probs_k)
            # Normalize to get probabilities
            probs.append(probs_k / np.sum(probs_k))

        return np.asarray(probs)
    
    def predict(self, X_test):
        """ 
        Predict the class labels for the given feature matrix.

        Args:
            X_test (array-like): Feature matrix for prediction.
        
        Returns:
            np.ndarray: Predicted class labels.
        """
        if not self.trained:
            raise Exception('Model is not trained.')
        
        probs = self.predict_proba(X_test)
        # Return the class with the highest probability
        return np.argmax(probs, axis=1)