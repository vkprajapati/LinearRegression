# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 03:00:14 2020

@author: ViShAl
"""


import pandas as pd
import numpy as np
import seaborn as sns

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv'
data = pd.read_csv(url, error_bad_lines=False)
data.describe(include='all')
data.isnull().sum().sort_values(ascending = True)

from sklearn.model_selection import train_test_split
train, test = train_test_split(data,test_size=0.25,random_state=40)
train.describe()


col_temp = ["T1","T2","T3","T4","T5","T6","T7","T8","T9"]
col_hum = ["RH_1","RH_2","RH_3","RH_4","RH_5","RH_6","RH_7","RH_8","RH_9"]
col_weather = ["T_out", "Tdewpoint","RH_out","Press_mm_hg", "Windspeed","Visibility"] 
col_light = ["lights"]
col_randoms = ["rv1", "rv2"]
col_target = ["Appliances"]


feature_vars = train[col_temp + col_hum + col_weather + col_light + col_randoms ]
target_vars = train[col_target]
feature_vars.describe()
feature_vars.lights.value_counts()
target_vars.describe()

_ = feature_vars.drop(['lights'], axis=1 , inplace= True) ;
feature_vars.head(2)

train_X = train[feature_vars.columns]
train_y = train[target_vars.columns]

test_X = test[feature_vars.columns]
test_y = test[target_vars.columns]

train_X.drop(["rv1","rv2","Visibility","T6","T9"],axis=1 , inplace=True)
test_X.drop(["rv1","rv2","Visibility","T6","T9"], axis=1, inplace=True)

train_X.columns
test_X.columns

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

train = train[list(train_X.columns.values) + col_target ]
test = test[list(test_X.columns.values) + col_target ]
sc_train = pd.DataFrame(columns=train.columns , index=train.index)
sc_train[sc_train.columns] = sc.fit_transform(train)
sc_test= pd.DataFrame(columns=test.columns , index=test.index)
sc_test[sc_test.columns] = sc.fit_transform(test)

sc_train.head()
sc_test.head()

train_X =  sc_train.drop(['Appliances'] , axis=1)
train_y = sc_train['Appliances']

test_X =  sc_test.drop(['Appliances'] , axis=1)
test_y = sc_test['Appliances']
train_X.head()
train_y.head()

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn import neighbors
from sklearn.svm import SVR
models = [
           ['Lasso: ', Lasso()],
           ['Ridge: ', Ridge()],
           ['KNeighborsRegressor: ',  neighbors.KNeighborsRegressor()],
           ['SVR:' , SVR(kernel='rbf')],
           ['RandomForest ',RandomForestRegressor()],
           ['ExtraTreeRegressor :',ExtraTreesRegressor()],
           ['GradientBoostingClassifier: ', GradientBoostingRegressor()] ,
           ['XGBRegressor: ', xgb.XGBRegressor()] ,
           ['MLPRegressor: ', MLPRegressor(  activation='relu', solver='adam',learning_rate='adaptive',max_iter=1000,learning_rate_init=0.01,alpha=0.01)]
         ]

import time
from math import sqrt
from sklearn.metrics import mean_squared_error

model_data = []
for name,curr_model in models :
    curr_model_data = {}
    curr_model.random_state = 78
    curr_model_data["Name"] = name
    start = time.time()
    curr_model.fit(train_X,train_y)
    end = time.time()
    curr_model_data["Train_Time"] = end - start
    curr_model_data["Train_R2_Score"] = metrics.r2_score(train_y,curr_model.predict(train_X))
    curr_model_data["Test_R2_Score"] = metrics.r2_score(test_y,curr_model.predict(test_X))
    curr_model_data["Test_RMSE_Score"] = sqrt(mean_squared_error(test_y,curr_model.predict(test_X)))
    model_data.append(curr_model_data)
    
model_data
df = pd.DataFrame(model_data)
df    

df.plot(x="Name", y=['Test_R2_Score' , 'Train_R2_Score' , 'Test_RMSE_Score'], kind="bar" , title = 'R2 Score Results' , figsize= (10,8)) ;



from sklearn import metrics
from sklearn.linear_model import LinearRegression
linearModel = LinearRegression()
linearModel.fit(train_X, train_y)
predictedValues = linearModel.predict(test_X)
print(linearModel.intercept_)
print(linearModel.coef_)

from sklearn.metrics import mean_absolute_error
mn = mean_absolute_error(test_y,curr_model.predict(test_X))
round(mn, 3)
ResidualSumSquares = np.sum(np.square(y_test - predictedValues))
round(ResidualSumSquares, 3)

from sklearn.metrics import mean_squared_error
RootMeanSquareError = np.sqrt(mean_squared_error(test_y,curr_model.predict(test_X)))
round(RootMeanSquareError, 3)

from sklearn.metrics import r2_score
r2_score = r2_score(test_y,curr_model.predict(test_X))
round(r2_score, 3)

from sklearn.linear_model import Ridge
RidgeRegression = Ridge(alpha=0.5)
RidgeRegression.fit(train_X, train_y)

from sklearn.linear_model import Lasso
LassoRegression = Lasso(alpha=0.001)
LassoRegression.fit(train_X, train_y)























