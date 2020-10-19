
import numpy as np
import functions as fun
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# #############################################################################
# Fit Model Manually
# #############################################################################
# y = 4 + 3x + N(1, 1) --------------------------------------------------------
points = 500
X = 2 * np.random.randn(points, 1)
y = 4 + 3 * X + np.random.randn(points, 1)
X_b = np.c_[np.ones((points, 1)), X]
# Fit the parameters ----------------------------------------------------------
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# Test ------------------------------------------------------------------------
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()
# #############################################################################
# Auto Fit
# #############################################################################
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
lin_reg.predict(X_new)
# #############################################################################
# Learning Curves
# #############################################################################
lin_reg = LinearRegression()
fun.plot_learning_curves(lin_reg, X, y)




polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])
fun.plot_learning_curves(polynomial_regression, X, y)
