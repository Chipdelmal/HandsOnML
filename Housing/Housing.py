
import numpy as np
import pandas as pd
import functions as fun
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


TEST_RATIO = .2
# #############################################################################
# Fetch and load
# #############################################################################
fun.fetch_housing_data()
housing = fun.load_housing_data()

# #############################################################################
# Get info and explore
# #############################################################################
housing.head()
housing.info()
housing.describe()

housing.get('ocean_proximity').value_counts()

housing.hist(bins=50, figsize=(20, 15))
plt.show()


# #############################################################################
# Split into training and test
# #############################################################################
(train_set, test_set) = fun.split_train_test(housing, TEST_RATIO)
# Use row index as id ---------------------------------------------------------
housing_with_id = housing.reset_index()
(train_set, test_set) = fun.split_train_test_by_id(
        housing_with_id, TEST_RATIO, 'index'
    )
# Use geolocation as id -------------------------------------------------------
housing_with_id['id'] = (housing['longitude'] * 1000) + (housing['latitude'])
(train_set, test_set) = fun.split_train_test_by_id(
        housing_with_id, TEST_RATIO, 'id'
    )
# ScikitLearn -----------------------------------------------------------------
(train_set, test_set) = train_test_split(
        housing, test_size=TEST_RATIO, random_state=134657281
    )
# Stratified ------------------------------------------------------------------
housing['income_cat'] = pd.cut(
        housing['median_income'],
        bins=[0, 1.5, 3, 4.5, 6, np.inf],
        labels=[1, 2, 3, 4, 5]
    )
# housing['income_cat'].hist()
split = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_RATIO, random_state=753481
    )
for (train_index, test_index) in split.split(housing, housing['income_cat']):
    (strat_train_set, strat_test_set) = (
            housing.loc[train_index], housing.loc[test_index]
        )
strat_test_set["income_cat"].value_counts() / len(strat_test_set)
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# #############################################################################
# Visualize data
# #############################################################################
