
import numpy as np
from scipy import stats
from compress_pickle import load
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

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
(housing_prepared, housing_labels, housing, num_attribs, cat_attribs, strat_test_set) = (
        rawData['prepared'], rawData['labels'], rawData['original'],
        rawData['num'], rawData['cat'], rawData['test']
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
# #############################################################################
# Analyze the Models and Errors
# #############################################################################
feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
# #############################################################################
# Final evaluation (against test dataset)
# #############################################################################
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(
        stats.t.interval(confidence, len(squared_errors) - 1,
        loc=squared_errors.mean(),
        scale=stats.sem(squared_errors))
    )
