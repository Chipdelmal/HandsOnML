
import os
import joblib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_openml


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


def dwldAndLoad(pth, dataset, version=1):
    datasetExists = os.path.exists(pth)
    if (not datasetExists):
        mnist = fetch_openml('mnist_784', version=version)
        joblib.dump(mnist, pth)
    else:
        mnist = joblib.load(pth)
    return mnist


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')


def pklFitModel(pth, model, X_train, y_train, OVW=False):
    mdlExists = os.path.exists(pth)
    if (not mdlExists) or (OVW):
        fit = model.fit(X_train, y_train)
        joblib.dump(fit, pth)
    else:
        fit = joblib.load(pth)
    return fit


def pklEval(pth, model, fun, X_train, y_train, OVW=False, **kwargs):
    mdlExists = os.path.exists(pth)
    if (not mdlExists) or (OVW):
        dta = fun(model, X_train, y_train, **kwargs)
        joblib.dump(dta, pth)
    else:
        dta = joblib.load(pth)
    return dta


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=mpl.cm.binary, interpolation="nearest")
    plt.axis("off")
