
import numpy as np
from compress_pickle import load
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

# #############################################################################
# Load Dataset
# #############################################################################
rawData = load(
        './processed/preHousing',
        compression=None, set_default_extension=False
    )
full_pipeline = load(
        './processed/preTransform',
        compression=None, set_default_extension=False
    )
(housing_prepared, housing_labels, housing) = (
        rawData['prepared'], rawData['labels'], rawData['original']
    )
# #############################################################################
# Grid Search
# #############################################################################
param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {
            'bootstrap': [False],
            'n_estimators': [3, 10],
            'max_features': [2, 3, 4]
        }
    ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(
        forest_reg, param_grid, cv=5,
        scoring='neg_mean_squared_error',
        return_train_score=True
    )
grid_search.fit(housing_prepared, housing_labels)
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
# #############################################################################
# Randomized Search
# #############################################################################
forest_reg = RandomForestRegressor()
random_search = RandomizedSearchCV(
        forest_reg, param_grid, cv=5,
        scoring='neg_mean_squared_error',
        return_train_score=True, n_iter=10
    )
grid_search.fit(housing_prepared, housing_labels)
