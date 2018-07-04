import pandas as pd
import numpy as np
import sklearn as sk

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import xgboost as xgb


traindata = pd.read_csv('train.csv')
testdata = pd.read_csv('test.csv')
data = pd.concat([traindata, testdata], ignore_index=True, sort=False)
# print(data.columns)
# print(data.head(5))
# print(data.dtypes)
# print(data.describe())
# print(data.isnull().sum())

columns_of_interest = ['Id', 'SalePrice', 'MSSubClass', 'LotArea', 'BldgType',
                'HouseStyle', 'OverallQual', 'OverallCond', 'GarageArea', 'PoolArea',
                'YearBuilt', '1stFlrSF', '2ndFlrSF',
                'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                'MiscVal', 'SaleType', 'SaleCondition']
datafeatures = data[columns_of_interest]
# print(dataFeatures.head(5))
# print(dataFeatures.describe())
datafeaturesOHE = pd.get_dummies(datafeatures)
# print(datafeaturesOHE.isnull().sum())

train = datafeaturesOHE.loc[datafeaturesOHE['SalePrice'].notna()]
test = datafeaturesOHE.loc[datafeaturesOHE['SalePrice'].isna()]

sale_price = train['SalePrice']

train = train.drop(['Id', 'SalePrice'], axis=1)
train_norm = train/train.max()

# Calculate train test split
X_train, X_val, y_train, y_val = train_test_split(train_norm, sale_price, random_state=0)
# X_train, X_val, y_train, y_val = train_test_split(datafeatures_norm, sale_price)

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
ind_params = {'seed':0,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'n_jobs': 4}
cv_params = {'learning_rate': [0.01, 0.1],
             'n_estimators': [500, 1000, 2000, 4000],
             'max_depth': [3, 5, 8],
             'min_child_weight': [1, 3, 5]}

xgbModel = XGBRegressor(**ind_params)

xgbGrid = GridSearchCV(xgbModel, param_grid=cv_params)
xgbGrid.fit(X_train, y_train)

n_estimatorsXGB_value = xgbGrid.best_estimator_.n_estimators
learning_rateXGB_value = xgbGrid.best_estimator_.learning_rate
max_depthXGB_value = xgbGrid.best_estimator_.max_depth
min_child_weightXGB_value = xgbGrid.best_estimator_.min_child_weight

n_estimatorsXGB_value = 1000
learning_rateXGB_value = 0.01
max_depthXGB_value = 5
min_child_weightXGB_value = 1

print("Best n_estimators: %d" % (n_estimatorsXGB_value))
print("best learning_rate: %f" % (learning_rateXGB_value))
print("Best max_depth: %d" % (max_depthXGB_value))
print("Best min_child_weight: %d" % (min_child_weightXGB_value))

housePriceModelXGB = XGBRegressor(n_estimators=n_estimatorsXGB_value,
                                  learning_rate=learning_rateXGB_value,
                                  max_depth=max_depthXGB_value,
                                  min_child_weight=min_child_weightXGB_value)

housePriceModelXGB.fit(X_train, y_train, early_stopping_rounds=5,
             eval_set=[(X_val, y_val)], verbose=False)

predictions = housePriceModelXGB.predict(X_val)
print(mean_absolute_error(y_val, predictions))

# Scoring the competition
logRMSE = np.sqrt(mean_squared_error(np.log(y_val), np.log(predictions)))

print("Score: %f" % (logRMSE))

IdDF = test['Id']
IdDF = IdDF.reset_index(drop=True)

test = test.drop(['Id', 'SalePrice'], axis=1)
test_norm = test/test.max()

testPredictions = housePriceModelXGB.predict(test_norm)
testPredictionsDF = pd.DataFrame(testPredictions, columns=['SalePrice'])
result = pd.concat([IdDF, testPredictionsDF], axis=1, sort=False)

result.to_csv('submission.csv', index=False)
