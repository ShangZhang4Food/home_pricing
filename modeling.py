"""
Created on Sun Oct 10 21:31:33 2021

Resource: Will Koehrsen
URL: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74 

@author: ShangZhang4Food
"""

import pandas as pd
import xgboost as xgb
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error


##### Preparation #####
# Import the data from csv file
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
df = pd.read_csv(__location__ + '/train_cleaned.csv')

# retrieve the array
data = df.values

# split into input and output elements
X, y = data[:, :-1], data[:, -1]

# split into train and test sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=1)

# summarize the shape of the train and test sets
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)



##### Modeling #####
LR = LinearRegression() 
RF=RandomForestRegressor(n_estimators = 200)
KNN=KNeighborsRegressor(n_neighbors=8)
DT=DecisionTreeRegressor(random_state=1)
xg=xgb.XGBRegressor(learning_rate=.1,booster='dart')

LR.fit(X_train,y_train)
y_pred=LR.predict(X_valid)
print('r2_score for RL: ',r2_score(y_valid,y_pred))
print('rmse for RL_model: ',np.sqrt(mean_squared_error(y_valid,y_pred)))

KNN.fit(X_train,y_train)
y_pred=KNN.predict(X_valid)
print('\nr2_score for KNN: ',r2_score(y_valid,y_pred))
print('rmse for KNN_model: ',np.sqrt(mean_squared_error(y_valid,y_pred)))

xg.fit(X_train,y_train)
y_pred=xg.predict(X_valid)
print('\nr2_score for xg: ',r2_score(y_valid,y_pred))
print('rmse for xg_model: ',np.sqrt(mean_squared_error(y_valid,y_pred)))

RF.fit(X_train,y_train)
y_pred=RF.predict(X_valid)
print('\nr2_score for RF: ',r2_score(y_valid,y_pred)) 
print('rmse for RF_model: ',np.sqrt(mean_squared_error(y_valid,y_pred)))

DT.fit(X_train,y_train)
y_pred=DT.predict(X_valid)
print('\nr2_score for DT: ',r2_score(y_valid,y_pred))
print('rmse for DT_model: ',np.sqrt(mean_squared_error(y_valid,y_pred)))



##### Pick a model to tune hyperparameters #####
xg2=xgb.XGBRegressor(learning_rate=.1,booster='dart', tree_method = 'approx', max_depth = 5)
xg2.fit(X_train,y_train)
y_pred=xg2.predict(X_valid)
print('\nr2_score for tuned XG: ',r2_score(y_valid,y_pred))
print('rmse for tuned XG: ',np.sqrt(mean_squared_error(y_valid,y_pred)))



##### Prediction #####
test_df = pd.read_csv(__location__ + '/test_cleaned.csv')
X_test = test_df.values
y_pred2 = xg2.predict(X_test)
