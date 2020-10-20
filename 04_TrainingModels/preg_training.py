
import numpy as np
import functions as fun
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * (X ** 2) + X + 2 + np.random.rand(m, 1)
# #############################################################################
# Training a degree 2 model
# #############################################################################
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
(lin_reg.coef_, lin_reg.intercept_)
# #############################################################################
# Learning curve of different regressions
# #############################################################################
fun.plot_learning_curves(lin_reg, X, y)
polynomial_regression = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression())
])
fun.plot_learning_curves(polynomial_regression, X, y, yRange=(0,1))
