import pandas as pd
import numpy as np

import os

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.python.keras.constraints import MaxNorm

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


""" Data preparation """
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

X = train.values
Y = sale_price.values

""" Neural network """
def larger_model():
    # create model
    model = Sequential()
    model.add(Dense(43, input_dim=43, kernel_initializer='normal',
                                 activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

def largerCustom_model():
    # create model
    model = Sequential()
    model.add(Dense(1024, input_dim=43, kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# # evaluate model with larger model
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50,
#             batch_size=5, verbose=1)))
# pipelineLarger = Pipeline(estimators)
#
# kfold = KFold(n_splits=10, random_state=seed)
# resultsLarger = cross_val_score(pipelineLarger, X, Y, cv=kfold, n_jobs=-1)

# evaluate model with a larger custom model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=largerCustom_model,
            verbose=1)))
pipelineLargerC = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)

# define the grid search parameters
neurons = [6]
batch_size = [1]
epochs = [100]
param_grid = dict(mlp__neurons=neurons, mlp__batch_size=batch_size, mlp__epochs=epochs)
grid = GridSearchCV(pipelineLargerC, param_grid=param_grid, cv=kfold, n_jobs=3)
grid_result = grid.fit(X, Y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# The mean absolute error of the 'larger' model from the tutorial
# print("Larger: %.2f (%.2f) MAE" % (resultsLarger.mean(), resultsLarger.std()))

# resultsLargerC = cross_val_score(pipelineLargerC, X, Y, cv=kfold, n_jobs=2)


# The larger model was the best so I built further on that
# print("LargerC: %.2f (%.2f) MAE" % (resultsLargerC.mean(), resultsLargerC.std()))

# # Scoring the competition
# logRMSE = np.sqrt(mean_squared_error(np.log(y_val), np.log(predictions)))
#
# print("Score: %f" % (logRMSE))
#
# IdDF = test['Id']
# IdDF = IdDF.reset_index(drop=True)
#
# test = test.drop(['Id', 'SalePrice'], axis=1)
# test_norm = test/test.max()
#
# testPredictions = housePriceModelXGB.predict(test_norm)
# testPredictionsDF = pd.DataFrame(testPredictions, columns=['SalePrice'])
# result = pd.concat([IdDF, testPredictionsDF], axis=1, sort=False)
#
# result.to_csv('submission.csv', index=False)

# Current result: 0.20714, place 4312, top 88%
# """
