import numpy as np

class GenerativeModel:
    """Base Class for Generative Models."""

    def __init__(self):
        self.K = None
        self.trained = False
    
    def fit(self, X_train, y_train):
        """
        Train the generative model.

        Args:
            X_train (array-like): Feature matrix for training.
            y_train (array-like): Target labels for training.
        """
        raise NotImplementedError("Subclass of GenerativeModel must implement this method")
    
    def predict(self, X_test):
        """
        Predict target labels for the given feature matrix.

        Args:
            X_test (array-like): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted labels.
        """
        raise NotImplementedError("Subclass of GenerativeModel must implement this method")
    
    def score(self, X_test, y_test):
        """
        Compute the score of the model using accuracy metric.
        
        Args:
            X_test (array-like): Feature matrix for testing.
            y_test (array-like): True labels for comparing.
            
        Returns:
            float: accuracy (proportion of correct predictions). 
        """
        if not self.trained:
            raise Exception('Model is not trained.')
        y_pred = self.predict(X_test)
        return np.mean(y_pred == y_test) 