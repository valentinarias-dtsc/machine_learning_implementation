import numpy as np
from generative_model import GenerativeModel

class NaiveBayes(GenerativeModel):
    """Naive Bayes Model."""

    def fit(self, X_train, y_train, model_type='gaussian', alpha=1.0):
        """ 
        Fit the Naive Bayes model to the training data.

        Args:
            X_train (array-like): Feature matrix for training.
            y_train (array-like): Target labels for training.
        """
        # Convert inputs to numpy arrays
        X = np.asarray(X_train)
        y = np.asarray(y_train)

        self.model_type = model_type

        # Get unique class labels
        self.K = np.unique(y)
        self.priors = {k: np.sum(y == k) / len(y) for k in self.K}

        # Initialize parameters based on model type
        if self.model_type.upper() == 'GAUSSIAN':
            self.mu = {k: np.mean(X[y==k], axis=0) for k in self.K}
            self.sigma = {k: np.var(X[y==k], axis=0) + 1e-9 for k in self.K} # Adding small value for numerical stability

        elif self.model_type.upper() == "BERNOULLI":
            self.cond_probs = {}
            for k in self.K:
                probs = (np.sum(X[y==k], axis=0) + alpha) / (len(X[y==k]) + 2 * alpha)
                self.cond_probs[k] = np.clip(probs, 1e-9, 1 - 1e-9) # Clipping to avoid log(0) issues

        elif self.model_type.upper() == "MULTINOMIAL":
            self.cond_probs = {}
            for k in self.K:
                probs = np.sum(X[y==k], axis=0) + alpha
                probs /= np.sum(probs)
                self.cond_probs[k] = np.clip(probs, 1e-9, 1 - 1e-9) # Clipping to avoid log(0) issues
        else: 
            raise ValueError(f'Invalid Model Type: {self.model_type}')
        
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
        log_probs = np.zeros((X.shape[0], len(self.K)))

        for i, k in enumerate(self.K):
            log_prior = np.log(self.priors[k])
            # Compute log likelihood based on model type
            if self.model_type.upper() == 'GAUSSIAN':
                mu_k = self.mu[k]
                sigma_k = self.sigma[k]
                # Log density function for Gaussian
                log_ll = np.sum(-0.5 * np.log(2 * np.pi * sigma_k) - ((X - mu_k) ** 2) / (2 * sigma_k), axis=1)
            elif self.model_type.upper() == 'BERNOULLI':
                prob_k = self.cond_probs[k]
                # Log likelihood for Bernoulli
                log_ll = np.sum(X * np.log(prob_k) + (1 - X) * np.log(1 - prob_k), axis=1)
            elif self.model_type.upper() == 'MULTINOMIAL':
                prob_k = self.cond_probs[k]
                # Log likelihood for Multinomial
                log_ll = np.dot(X, np.log(prob_k))

            log_probs[:, i] = log_prior + log_ll

        # Normalize log probabilities
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        return probs / probs.sum(axis=1, keepdims=True)

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

        X = np.asarray(X_test)
        probs = self.predict_proba(X)
        return self.K[np.argmax(probs, axis=1)]