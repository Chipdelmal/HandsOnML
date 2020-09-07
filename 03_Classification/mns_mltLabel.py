
import os
import numpy as np
import functions as fun
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict


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
(y_train, y_test) = [y_st.astype(np.uint8) for y_st in (y_train, y_test)]
# Setting up "large" and "odd" labels -----------------------------------------
(y_train_large, y_train_odd) = ((y_train >= 7), (y_train % 2 == 1))
y_multilabel = np.c_[y_train_large, y_train_odd]
# #############################################################################
# KNN
# #############################################################################
knn_clf = KNeighborsClassifier()
knn_clf = fun.pklFitModel(modFld+'knn_clf.pkl', knn_clf, X_train, y_multilabel)
knn_clf.predict([X[0]])
knn_prd = fun.pklEval(
        modFld+'knn_clf_pred.pkl', knn_clf, cross_val_predict,
        X_train, y_multilabel, OVW=False, cv=3
    )
