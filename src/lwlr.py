import numpy as np
from numpy.linalg import inv, norm

def LWLR(X_train, y_train, x_query, tau):
    X = np.asarray(X_train)
    y = np.asarray(y_train)
    x = np.asarray(x_query).reshape(1, -1)

    weights = np.exp(- (X - x) ** 2 / (2 * tau ** 2))
    W = np.diagflat(weights)

    X = np.hstack([np.ones((X.shape[0], 1)), X])
    x = np.concatenate(([1], x_query))

    XTWy = X.T @ W @ y
    XTWX = X.T @ W @ X
    XTWX += np.eye(XTWX.shape[0]) * 1e-6
    theta = inv(XTWX) @ XTWy

    return float(x @ theta)

def LW_LinReg(X_train, y_train, X_test, tau):
    return np.array([LWLR(X_train, y_train, x, tau) for x in X_test])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def LWlR(X_train, y_train, x_query, tau, max_iter=20, lr=0.01, tol=1e-6):
    
    diff = X_train - x_query
    dists = np.sum(diff**2, axis=1)
    weights = np.exp(- dists / (2 * tau**2))
    W = np.diag(weights)

    X = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    x = np.concatenate(([1], x_query)).reshape(1, -1)
    y = np.asarray(y_train)

    theta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        h = sigmoid(X @ theta)
        grad = X.T @ (W @ (h - y))
        old_theta = theta.copy()
        theta -= lr * grad
        if norm(theta - old_theta) < tol:
            break
    prob = sigmoid(x @ theta)
    return (prob >= .5).astype(int)

def LW_LogReg(X_train, y_train, X_test, tau, max_iter=20, lr=0.01, tol=1e-6):
    return np.array([LWlR(X_train, y_train, x, tau, max_iter, lr, tol) for x in X_test])