
import os.path
import numpy as np
import functions as fun
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score

# #############################################################################
# Download dataset
# #############################################################################
datasetExists = os.path.exists('./datasets/mnist.pkl')
mnist = fun.dwldAndLoad('./datasets/mnist.pkl', 'mnist_784', 1)
# #############################################################################
# Examine dataset
# #############################################################################
(X, y) = (mnist["data"], mnist["target"])
(imgN, imgSz) = X.shape
i = 10
some_digit = X[i]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap="binary")
y[i]
y = y.astype(np.uint8)
# #############################################################################
# Split Dataset (it's already pre-split)
# #############################################################################
sIx = 60000
(X_train, X_test, y_train, y_test) = (X[:sIx], X[sIx:], y[:sIx], y[sIx:])
# #############################################################################
# Training a binary 5 classifier
# #############################################################################
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
# Predict ---------------------------------------------------------------------
i = 10
some_digit = X[i]
sgd_clf.predict([some_digit])
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap="binary")
# #############################################################################
# Cross-Validation
# #############################################################################
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# Dumb "never a 5" classifier -------------------------------------------------
never_5_clf = fun.Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# Confusion matrix ------------------------------------------------------------
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
confusion_matrix(y_train_5, y_train_pred)
# Precision and Recall --------------------------------------------------------
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)
f1_score(y_train_5, y_train_pred)
# #############################################################################
# Precision/Recall
# #############################################################################
i = 10
some_digit = X[i]
y_scores = sgd_clf.decision_function([some_digit])
y_scores
y_scores = cross_val_predict(
        sgd_clf, X_train, y_train_5, cv=3, method="decision_function"
    )
(precisions, recalls, thresholds) = precision_recall_curve(y_train_5, y_scores)
fun.plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
y_train_pred_90 = (y_scores >= threshold_90_precision)
precision_score(y_train_5, y_train_pred_90)
recall_score(y_train_5, y_train_pred_90)
y_train_5
