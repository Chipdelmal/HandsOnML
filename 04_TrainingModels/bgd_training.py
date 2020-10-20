

import numpy as np
import functions as fun
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor


(eta, n_iterations, m) = (.1, 1000, 100)
# #############################################################################
# Simmed Data
# #############################################################################
# y = 4 + 3x + N(1, 1) --------------------------------------------------------
points = 500
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)
X_b = np.c_[np.ones((m, 1)), X]
# #############################################################################
# Batch Gradient Descent
# #############################################################################
theta = np.random.randn(2, 1)  # random initialization
for iteration in range(n_iterations):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
theta
# #############################################################################
# Stochastic Gradient Descent
# #############################################################################
n_epochs = 50
(t0, t1) = (5, 50)
# Manual Learn ----------------------------------------------------------------
theta = np.random.rand(2, 1)
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:(random_index + 1)]
        yi = y[random_index:(random_index + 1)]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = fun.learning_schedule(epoch * m + i, t0, t1)
        theta = theta - eta * gradients
theta
# SciKit Learn ----------------------------------------------------------------
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())
sgd_reg.intercept_, sgd_reg.coef_
