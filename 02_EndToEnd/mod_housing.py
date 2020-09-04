
import joblib
import numpy as np
import functions as fun
from compress_pickle import load
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

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
# Create a Linear Regression Model
# #############################################################################
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
# Evaluate --------------------------------------------------------------------
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))
# Calculate RMSE --------------------------------------------------------------
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
# #############################################################################
# Create a Decision Tree Regression Model
# #############################################################################
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
# #############################################################################
# Cross Validation
# #############################################################################
scores = cross_val_score(
        tree_reg, housing_prepared, housing_labels,
        scoring="neg_mean_squared_error", cv=10
    )
tree_rmse_scores = np.sqrt(-scores)
fun.display_scores(tree_rmse_scores)
# #############################################################################
# Create a Random Forests Model
# #############################################################################
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
scores = cross_val_score(
        forest_reg, housing_prepared, housing_labels,
        scoring="neg_mean_squared_error", cv=10
    )
forest_rsme_scores = np.sqrt(-scores)
fun.display_scores(forest_rsme_scores)
# #############################################################################
# Save Models
# #############################################################################
joblib.dump(forest_reg, "./models/forest.pkl")
