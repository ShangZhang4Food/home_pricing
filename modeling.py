<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 21:31:33 2021

@author: jerry
"""

import pandas as pd
import xgboost as xgb
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import r2_score,mean_squared_error

# Import the data from csv file
df = pd.read_csv("train_cleaned.csv")

# retrieve the array
data = df.values

# split into input and output elements
X, y = data[:, :-1], data[:, -1]

# split into train and test sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=1)

# summarize the shape of the train and test sets
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

LR_model=LinearRegression()   
RF_model=RandomForestRegressor(n_estimators=200)
KNN_model=KNeighborsRegressor(n_neighbors=8)
DT_model=DecisionTreeRegressor(random_state=1)
xg_model=xgb.XGBRegressor(learning_rate=.1,booster='dart')

LR_model.fit(X_train,y_train)
y_pred=LR_model.predict(X_valid)
print('r2_score for RL: ',r2_score(y_valid,y_pred))
print('rmse for RL_model: ',np.sqrt(mean_squared_error(y_valid,y_pred)))

KNN_model.fit(X_train,y_train)
y_pred=KNN_model.predict(X_valid)
print('\nr2_score for KNN: ',r2_score(y_valid,y_pred))
print('rmse for KNN_model: ',np.sqrt(mean_squared_error(y_valid,y_pred)))

xg_model.fit(X_train,y_train)
y_pred=xg_model.predict(X_valid)
print('\nr2_score for xg: ',r2_score(y_valid,y_pred))
print('rmse for xg_model: ',np.sqrt(mean_squared_error(y_valid,y_pred)))

RF_model.fit(X_train,y_train)
y_pred=RF_model.predict(X_valid)
print('\nr2_score for RF: ',r2_score(y_valid,y_pred)) 
print('rmse for RF_model: ',np.sqrt(mean_squared_error(y_valid,y_pred)))

DT_model.fit(X_train,y_train)
y_pred=DT_model.predict(X_valid)
print('\nr2_score for DT: ',r2_score(y_valid,y_pred))
=======
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 21:31:33 2021

@author: jerry
"""

import pandas as pd
import xgboost as xgb
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import r2_score,mean_squared_error

# Import the data from csv file
df = pd.read_csv("train_cleaned.csv")

# retrieve the array
data = df.values

# split into input and output elements
X, y = data[:, :-1], data[:, -1]

# split into train and test sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=1)

# summarize the shape of the train and test sets
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

LR_model=LinearRegression()   
RF_model=RandomForestRegressor(n_estimators=200)
KNN_model=KNeighborsRegressor(n_neighbors=8)
DT_model=DecisionTreeRegressor(random_state=1)
xg_model=xgb.XGBRegressor(learning_rate=.1,booster='dart')

LR_model.fit(X_train,y_train)
y_pred=LR_model.predict(X_valid)
print('r2_score for RL: ',r2_score(y_valid,y_pred))
print('rmse for RL_model: ',np.sqrt(mean_squared_error(y_valid,y_pred)))

KNN_model.fit(X_train,y_train)
y_pred=KNN_model.predict(X_valid)
print('\nr2_score for KNN: ',r2_score(y_valid,y_pred))
print('rmse for KNN_model: ',np.sqrt(mean_squared_error(y_valid,y_pred)))

xg_model.fit(X_train,y_train)
y_pred=xg_model.predict(X_valid)
print('\nr2_score for xg: ',r2_score(y_valid,y_pred))
print('rmse for xg_model: ',np.sqrt(mean_squared_error(y_valid,y_pred)))

RF_model.fit(X_train,y_train)
y_pred=RF_model.predict(X_valid)
print('\nr2_score for RF: ',r2_score(y_valid,y_pred)) 
print('rmse for RF_model: ',np.sqrt(mean_squared_error(y_valid,y_pred)))

DT_model.fit(X_train,y_train)
y_pred=DT_model.predict(X_valid)
print('\nr2_score for DT: ',r2_score(y_valid,y_pred))
>>>>>>> e61450b927a7dd8c75898aae1b16ca7ff005a3d5
print('rmse for DT_model: ',np.sqrt(mean_squared_error(y_valid,y_pred)))