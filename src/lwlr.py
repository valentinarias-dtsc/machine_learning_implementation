import numpy as np
from numpy.linalg import inv, norm

def LWLR(X_train, y_train, x_query, tau):
    """Locally Weighted Linear Regression (LWLR) for a single query point.
    Args:
        X_train (array-like): Training feature data.
        y_train (array-like): Training target values.
        x_query (array-like): Query point for prediction.
        tau (float): Bandwidth parameter controlling the locality."""
    # Convert inputs to numpy arrays
    X = np.asarray(X_train)
    y = np.asarray(y_train)
    x = np.asarray(x_query).reshape(1, -1)

    # Calculate weights based on the distance from the query point
    weights = np.exp(- (X - x) ** 2 / (2 * tau ** 2))
    W = np.diagflat(weights)

    # Add a bias term to the feature matrix
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    x = np.concatenate(([1], x_query))

    # Compute the locally weighted linear regression parameters
    XTWy = X.T @ W @ y
    XTWX = X.T @ W @ X
    XTWX += np.eye(XTWX.shape[0]) * 1e-6 # Regularization to avoid singular matrix
    theta = inv(XTWX) @ XTWy

    # Return the prediction for the query point
    return float(x @ theta)

def LW_LinReg(X_train, y_train, X_test, tau):
    """Locally Weighted Linear Regression (LWLR) for multiple query points.
    Args:
        X_train (array-like): Training feature data.
        y_train (array-like): Training target values.
        X_test (array-like): Query points for prediction.
        tau (float): Bandwidth parameter controlling the locality."""
    X_test = np.asarray(X_test)
    return np.array([LWLR(X_train, y_train, x, tau) for x in X_test])

def sigmoid(z):
    """Sigmoid function for logistic regression."""
    z = np.asarray(z)
    return 1 / (1 + np.exp(-z))

def LWlR(X_train, y_train, x_query, tau, max_iter=20, lr=0.01, tol=1e-6):
    """Locally Weighted Logistic Regression (LWlR) for a single query point.
    Args:
        X_train (array-like): Training feature data.
        y_train (array-like): Training target values.
        x_query (array-like): Query point for prediction.
        tau (float): Bandwidth parameter controlling the locality.
        max_iter (int): Maximum number of iterations for gradient descent.
        lr (float): Learning rate for gradient descent.
        tol (float): Tolerance for convergence."""
    
    # Calculate weights based on the distance from the query point
    diff = X_train - x_query
    dists = np.sum(diff**2, axis=1)
    weights = np.exp(- dists / (2 * tau**2))
    W = np.diag(weights)

    # Add a bias term to the feature matrix and prepare the query point
    X = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    x = np.concatenate(([1], x_query)).reshape(1, -1)
    y = np.asarray(y_train)

    # Initialize parameters for gradient descent
    theta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        # Compute the gradient and update theta
        h = sigmoid(X @ theta)
        grad = X.T @ (W @ (h - y))
        old_theta = theta.copy()
        theta -= lr * grad
        # Check for convergence
        if norm(theta - old_theta) < tol:
            break
    # Compute the probability and return the predicted class
    prob = sigmoid(x @ theta)
    return (prob >= .5).astype(int)

def LW_LogReg(X_train, y_train, X_test, tau, max_iter=20, lr=0.01, tol=1e-6):
    """Locally Weighted Logistic Regression (LWlR) for multiple query points.
    Args:
        X_train (array-like): Training feature data.
        y_train (array-like): Training target values.
        X_test (array-like): Query points for prediction.
        tau (float): Bandwidth parameter controlling the locality.
        max_iter (int): Maximum number of iterations for gradient descent.
        lr (float): Learning rate for gradient descent.
        tol (float): Tolerance for convergence."""
    X_test = np.asarray(X_test)
    return np.array([LWlR(X_train, y_train, x, tau, max_iter, lr, tol) for x in X_test])