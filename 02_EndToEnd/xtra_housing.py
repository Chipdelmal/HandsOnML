
import joblib
from sklearn import svm
import functions as fun
from compress_pickle import load
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# #############################################################################
# Load Dataset
# #############################################################################
rawData = load(
        './processed/preHousing.pkl',
        compression=None, set_default_extension=False
    )
full_pipeline = load(
        './processed/preTransform.pkl',
        compression=None, set_default_extension=False
    )
(housing_prepared, housing_labels, housing, num_attribs, cat_attribs, strat_test_set) = (
        rawData['prepared'], rawData['labels'], rawData['original'],
        rawData['num'], rawData['cat'], rawData['test']
    )
# #############################################################################
# Fitting Model
# #############################################################################
mdl = svm.SVR()
mdl.fit(housing_prepared, housing_labels)
# Cross Validate --------------------------------------------------------------
scores = cross_val_score(
        mdl, housing_prepared, housing_labels,
        scoring="neg_mean_squared_error", cv=10
    )
# Evaluate --------------------------------------------------------------------
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", mdl.predict(some_data_prepared))
print("Labels:", list(some_labels))
# #############################################################################
# Grid Search
# #############################################################################
mdl = svm.SVR()
params = {
        'kernel': ['linear', 'poly', 'rbf'],
        'gamma': ['auto', 'scale'],
        'C': [.5, 1, 1.5]
    }
gs = GridSearchCV(
        mdl, params, cv=5,
        scoring='neg_mean_squared_error',
        return_train_score=True
    )
gs = fun.pklFitModel(
        './processed/gsSVM.pkl', gs,
        housing_prepared, housing_labels
    )
(pmsBst, mdlBst) = (gs.best_params_, gs.best_estimator_)
mdlBst
