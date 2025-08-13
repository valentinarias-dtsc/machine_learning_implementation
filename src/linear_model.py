import numpy as np

class LinearModel:
    """Base class for linear models"""

    def __init__(self, max_iter=100000, tol=1e-6, lr=1e-6, verbose=False, theta_0=None):
        """
        Args:
            max_iter (int): maximum number of iterations
            tol (float): tolerance for the convergence criterion
            lr (float): step size for each iteration
            verbose (bool): prints the loss values during training
            theta_0 (array-like): initialization for the parameters
        """

        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        self.verbose = verbose
        self.theta = theta_0
        self.trained = False
    
    def fit(self, X_train, y_train):
        """
        Fit the linear model.
        
        Args:
            X_train (array-like): Feature matrix for training.
            y_train (array-like): Target values for training.
        """

        raise NotImplementedError('Subclass of LinearModel must implement the fit method.')
    
    def predict(self, X_test):
        """
        Predict target values for the given feature matrix.

        Args:
            X_test (array-like): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        
        raise NotImplementedError('Subclass of LinearModel must implement the predict method.')
    
    def get_params(self):
        """
        Returns the parameters of the model.

        Returns:
            theta: parameters of the model
        """

        if self.trained:
            return self.theta
        else: raise Exception('Model is not trained.')
    
    def _score(self, X_test, y_test, metric='mse'):
        """
        Compute the score of the model using the specified metric.
    
        Args:
            X_test (array-like): Feature matrix for testing.
            y_test (array-like): True labels or values.
            metric (str): Metric to use ('mse', 'r2', 'accuracy').
    
        Returns:
            float: Computed score.
        """
        y_pred = self.predict(X_test)
        if metric == 'mse':
            return np.mean((y_pred - y_test) ** 2)
        elif metric == 'r2':
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            return 1 - ss_res / ss_tot
        elif metric == 'accuracy':
            return np.mean(y_pred == y_test)
        else:
            raise ValueError(f"Unknown metric: {metric}")
