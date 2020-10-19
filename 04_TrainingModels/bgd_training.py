

import numpy as np
import matplotlib.pyplot as plt

eta = .1
n_iterations = 1000
m = 100
# #############################################################################
# Simmed Data
# #############################################################################
# y = 4 + 3x + N(1, 1) --------------------------------------------------------
points = 500
X = 2 * np.random.randn(points, 1)
y = 4 + 3 * X + np.random.randn(points, 1)
X_b = np.c_[np.ones((points, 1)), X]
# #############################################################################
theta = np.random.randn(2, 1)
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
