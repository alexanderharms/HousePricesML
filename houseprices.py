import pandas as pd
import numpy as np
import sklearn as sk

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV

data = pd.read_csv('train.csv')
# print(data.columns)
# print(data.head(5))
# print(data.describe())

sale_price = data['SalePrice']
columns_of_interest = ['MSSubClass', 'LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
                'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd']
datafeatures = data[columns_of_interest]
datafeatures_norm = datafeatures/datafeatures.max()
# print(dataFeatures.head(5))
# print(dataFeatures.describe())

# Calculate train test split
X_train, X_val, y_train, y_val = train_test_split(datafeatures, sale_price, random_state=0)

# Define model
model = RandomForestRegressor()

# Grid Search
min_samples_splitList = [2, 5, 8, 10, 15, 20]
max_leaf_nodeList = [5, 50, 500, 5000]
n_estimatorsList = [2, 5, 8, 10, 15, 20]

grid = GridSearchCV(estimator=model, param_grid=dict(n_estimators=n_estimatorsList,
                    min_samples_split=min_samples_splitList,
                    max_leaf_nodes=max_leaf_nodeList))
grid.fit(X_train, y_train)
print(grid)
print(grid.best_score_)
print("Best max_leaf_nodes: %d" % (grid.best_estimator_.max_leaf_nodes))
print("Best min_samples_split: %d" % (grid.best_estimator_.min_samples_split))
print("Best n_estimators: %d" % (grid.best_estimator_.n_estimators))

n_estimators_value = grid.best_estimator_.n_estimators
min_samples_split_value = grid.best_estimator_.min_samples_split
max_leaf_nodes_value = grid.best_estimator_.max_leaf_nodes

# Define model
housePriceModelRF = RandomForestRegressor(n_estimators=n_estimators_value,
                        max_leaf_nodes=max_leaf_nodes_value,
                        min_samples_split=min_samples_split_value)

#Fit model
housePriceModelRF.fit(X_train, y_train)
# Predict
val_predictionsRF = housePriceModelRF.predict(X_val)
print(mean_absolute_error(y_val, val_predictionsRF))

# # XGBoost
# housePriceModelXGB = XGBRegressor(n_estimators=1000, learning_rate=0.01)
# housePriceModelXGB.fit(X_train, y_train, early_stopping_rounds=1,
#              eval_set=[(X_val, y_val)], verbose=False)
# predictions = housePriceModelXGB.predict(X_val)
# print(mean_absolute_error(y_val, predictions))
