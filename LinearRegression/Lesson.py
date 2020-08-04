# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 01:44:41 2020

@author: ViShAl
"""


import pandas as pd
import numpy as np
import seaborn as sns

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv'
data = pd.read_csv(url, error_bad_lines=False)
data.describe(include='all')

simpleLinearReg = data[['T2' , 'T6']].sample(15, random_state = 2)
simpleLinearReg 

























colName = {'T1':'Temperature_kitchen_area', 
           'T2':'Temperature_living_room_area', 
           'T3':'Temperature_laundry_room_area', 
           'T4':'Temperature_office_room',
           'T5':'Temperature_bathroom',
           'T6':'Temperature_outside_building',
           'T7':'Temperature_ironing_room',
           'T8':'Temperature_teenager_room_2',
           'T9':'Temperature_parents_room',
           'T_out ':'Temperature_outside',
           'RH_1':'Humidity_kitchen_area',
           'RH_2':'Humidity_living_room_area',
           'RH_3':'Humidity_laundry_room_area',
           'RH_4':'Humidity_office_room',
           'RH_5':'Humidity_bathroom',
           'RH_6':'Humidity_outside_building',
           'RH_7':'Humidity_ironing_room',
           'RH_8':'Humidity_teenager_room_2',
           'RH_9':'Humidity_parents_room',
           'RH_out':'Humidity_outside'}
data = data.rename(columns = colName)

x = (['Temperature_living_room_area'])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = data.drop(columns = ['date', 'lights'])
normalisedData = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
featuresData = normalisedData.drop(columns='Appliances')
target = normalisedData['Appliances']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(featuresData, target, test_size=0.7, random_state=42)


from sklearn import metrics
from sklearn.linear_model import LinearRegression
linearModel = LinearRegression()
linearModel.fit(x_train, y_train)
predictedValues = linearModel.predict(x_test)
print(linearModel.intercept_)
print(linearModel.coef_)

from sklearn.metrics import mean_absolute_error
mn = mean_absolute_error(y_test, predictedValues)
round(mn, 3)
ResidualSumSquares = np.sum(np.square(y_test - predictedValues))
round(ResidualSumSquares, 3)

from sklearn.metrics import mean_squared_error
RootMeanSquareError = np.sqrt(mean_squared_error(y_test, predictedValues))
round(RootMeanSquareError, 3)

from sklearn.metrics import r2_score
r2_score = r2_score(y_test, predictedValues)
round(r2_score, 3)

from sklearn.linear_model import Ridge
RidgeRegression = Ridge(alpha=0.5)
RidgeRegression.fit(x_train, y_train)

from sklearn.linear_model import Lasso
LassoRegression = Lasso(alpha=0.001)
LassoRegression.fit(x_train, y_train)

















