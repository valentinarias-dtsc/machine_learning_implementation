import numpy as np
from numpy.linalg import inv, det
from generative_model import GenerativeModel

class GDA(GenerativeModel):
    """Gaussian Discriminant Analysis Model for multi-class problems."""

    def fit(self, X_train, y_train):
        """
        Fit the GDA model to the training data.

        Args:
            X_train (array-like): Feature matrix for training samples.
            y_train (array-like): Target labels for training samples.
        """
        X = np.asarray(X_train)
        y = np.asarray(y_train)

        self.K = np.unique(y)  # Unique class labels
        reg = 1e-6 * np.eye(X.shape[1])  # Regularization for covariance
        self.priors = {}  # Class priors
        self.mu = {}      # Class means
        self.sigma = {}   # Class covariances
        for k in self.K:
            self.priors[k] = np.mean(y == k)
            self.mu[k] = np.mean(X[y == k], axis=0)
            self.sigma[k] = np.cov(X[y == k], rowvar=False) + reg
                
        self.trained = True

    def predict_proba(self, X_test):
        """
        Predict class probabilities for the given samples.

        Args:
            X_test (array-like): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted probabilities for each class.
        """
        if not self.trained:
            raise Exception('Model is not trained.')

        X = np.asarray(X_test)
        probs = []
        for x in X:
            probs_k = []
            for k in self.K:
                # Compute the Gaussian likelihood
                norm_const = 1 / np.sqrt(det(self.sigma[k]))
                exp_term = np.exp(-0.5 * ((x - self.mu[k]) @ inv(self.sigma[k])) @ (x - self.mu[k]).T)
                joint_prob = norm_const * exp_term * self.priors[k]
                probs_k.append(joint_prob)
            probs.append(probs_k)
        probs = np.asarray(probs)
        probs /= np.sum(probs, axis=1, keepdims=True)  # Normalize to get probabilities
        return probs

    def predict(self, X_test):
        """
        Predict class labels for the given samples.

        Args:
            X_test (array-like): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted class labels.
        """
        if not self.trained:
            raise Exception('Model is not trained.')
        
        probs = self.predict_proba(X_test)
        max_probs = np.argmax(probs, axis=1)
        return np.asarray([self.K[i] for i in max_probs])