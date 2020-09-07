
import os
import joblib
import numpy as np
import functions as fun
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
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
# #############################################################################
# SVM
# #############################################################################
# Auto (OvO) ------------------------------------------------------------------
svm_clf = SVC()
svm_clf = fun.pklFitModel(modFld+'svm_clf.pkl', svm_clf, X_train, y_train)
# Test ------------------------------------------------------------------------
some_digit = X[10]
svm_clf.predict([some_digit])
some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores
svm_clf.classes_
# One VS Rest -----------------------------------------------------------------
ovr_clf = OneVsRestClassifier(SVC())
ovr_clf = fun.pklFitModel(modFld+'ovr_clf.pkl', ovr_clf, X_train, y_train)
ovr_scr = fun.pklEval(
        modFld+'ovr_clf_scr.pkl', ovr_clf, cross_val_score,
        X_train, y_train, OVW=False,
        cv=3, scoring="accuracy"
    )
sgd_clf = OneVsRestClassifier(SGDClassifier())
sgd_clf = fun.pklFitModel(modFld+'sgd_clf.pkl', sgd_clf, X_train, y_train)
sgd_scr = fun.pklEval(
        modFld+'sgd_clf_scr.pkl', sgd_clf, cross_val_score,
        X_train, y_train, OVW=False,
        cv=3, scoring="accuracy"
    )
# #############################################################################
# Scaling inputs
# #############################################################################
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
sgd_scl = fun.pklEval(
        modFld+'sgd_clf_predScaled.pkl', sgd_clf, cross_val_score,
        X_train_scaled, y_train, OVW=False,
        cv=3, scoring="accuracy"
    )
# #############################################################################
# Error Analysis
# #############################################################################
y_train_pred = fun.pklEval(
        modFld+'sgd_clf_pred.pkl', sgd_clf, cross_val_predict,
        X_train_scaled, y_train, OVW=False, cv=3
    )
conf_mx = confusion_matrix(y_train, y_train_pred)
plt.matshow(conf_mx, cmap=plt.cm.gray)
norm_conf_mx = conf_mx / conf_mx.sum(axis=1, keepdims=True)
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
