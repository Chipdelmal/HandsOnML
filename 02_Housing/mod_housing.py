
import numpy as np
from compress_pickle import load
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

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
