
import numpy as np
import pandas as pd
import functions as fun
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from compress_pickle import dump
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
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
        bins=[0, 1.5, 3, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5]
    )
housing['income_cat'].hist()
split = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_RATIO, random_state=753481
    )
for (train_index, test_index) in split.split(housing, housing['income_cat']):
    (strat_train_set, strat_test_set) = (
            housing.loc[train_index], housing.loc[test_index]
        )
strat_test_set['income_cat'].value_counts() / len(strat_test_set)
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)
# #############################################################################
# Visualize data and look for patterns
# #############################################################################
housing.plot(
        kind='scatter', x='longitude', y='latitude', alpha=.4,
        s=housing['population']/100, label='population', figsize=(10, 7),
        c='median_house_value', cmap=plt.get_cmap('RdPu'), colorbar=True
    )
# Correlations ----------------------------------------------------------------
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)
attributes = [
        'median_house_value', 'median_income',
        'total_rooms', 'housing_median_age'
    ]
scatter_matrix(housing[attributes], figsize=(12, 8))
housing.plot(
        kind='scatter', x='median_income', y='median_house_value', alpha=.1
    )
# Adding variables ------------------------------------------------------------
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
# #############################################################################
# Data cleaning
# #############################################################################
# Separating labels from features ---------------------------------------------
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set.get('median_house_value').copy()
# Dealing with missing features -----------------------------------------------
housing.dropna(subset=['total_bedrooms'])       # Get rid of rows with NA
housing.drop('total_bedrooms', axis=1)          # Get rid of the attribute
median = housing['total_bedrooms'].median()     # Fill in with value
housing['total_bedrooms'].fillna(median, inplace=True)
# Sklearn imputer -------------------------------------------------------------
imputer = SimpleImputer(strategy='median')
housing_num = housing.drop('ocean_proximity', axis=1)  # Drop non-numerical
imputer.fit(housing_num)
imputer.statistics_
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(
        X, columns=housing_num.columns, index=housing_num.index
    )
# Categorical variables encoding ----------------------------------------------
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]
ordinal_encoder.categories_
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
housing_cat_1hot.toarray()
cat_encoder.categories_
# Custom transformer ----------------------------------------------------------
attr_adder = fun.CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
# #############################################################################
# Feature scaling and Pipelines
# #############################################################################
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('attribs_adder', fun.CombinedAttributesAdder()),
        ('std_scaler', StandardScaler())
    ])
housing_num_tr = num_pipeline.fit_transform(housing_num)
# Categorical and numerical ---------------------------------------------------
(num_attribs, cat_attribs) = (list(housing_num), ["ocean_proximity"])
full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs)
    ])
housing_prepared = full_pipeline.fit_transform(housing)
# #############################################################################
# Export data
# #############################################################################
obj = {
        'original': housing,
        'labels': housing_labels,
        'prepared': housing_prepared
    }
with open('./processed/preHousing', "wb") as f:
    dump(obj, f, compression=None, set_default_extension=False)
with open('./processed/preTransform', "wb") as f:
    dump(full_pipeline, f, compression=None, set_default_extension=False)
