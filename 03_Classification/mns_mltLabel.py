
import os
import joblib
import numpy as np
import functions as fun
from sklearn.neighbors import KNeighborsClassifier


# #############################################################################
# Download dataset
# #############################################################################
(modFld, dtaFld) = ('./models/', './datasets/')
datasetExists = os.path.exists('./datasets/mnist.pkl')
mnist = fun.dwldAndLoad('./datasets/mnist.pkl', 'mnist_784', 1)
# #############################################################################
# Split Dataset (it's already pre-split)
# #############################################################################
sIx = 60000
(X, y) = (mnist["data"], mnist["target"])
(X_train, X_test, y_train, y_test) = (X[:sIx], X[sIx:], y[:sIx], y[sIx:])
