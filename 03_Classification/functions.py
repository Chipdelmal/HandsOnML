
import os
import joblib
import numpy as np
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
    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
