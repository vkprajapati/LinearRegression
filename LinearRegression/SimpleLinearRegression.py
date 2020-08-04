# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 00:49:36 2020

@author: ViShAl
"""


import pandas as pd
import numpy as np
import seaborn as sns

url1 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx'
data1 = pd.read_excel(url1)


colName = {'X1':'Relative_Compactness', 'X2': 'Surface_Area', 'X3': 'Wall_Area', 'X4': 'Roof_Area', 'X5': 'Overall_Height', 'X6': 'Orientation', 'X7': 'Glazing_Area', 'X8': 'Glazing_Area_Distribution', 'Y1': 'Heating_Load', 'Y2': 'Cooling_Load'}

data = data.rename(columns = colName)

simpleLinearReg = data[['Relative_Compactness' , 'Cooling_Load']].sample(15, random_state = 2)

sns.regplot(x = "Relative_Compactness", y = "Cooling_Load", data = simpleLinearReg)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

normalisedData = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

featuresData = normalisedData.drop(columns=['Heating_Load', 'Cooling_Load'])

target = normalisedData['Heating_Load']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(featuresData, target, test_size=0.3, random_state=1)

from sklearn.linear_model import LinearRegression

linearModel = LinearRegression()
linearModel.fit(x_train, y_train)
predictedValues = linearModel.predict(x_test)

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

def getWeightsData(model, feat, col_name):
    weights = pd.Series(model.coef_, feat.columns).sort_values()
    weightsData = pd.DataFrame(weights).reset_index()
    weightsData.columns = ['Features', col_name]
    weightsData[col_name].round(3)
    return weightsData

linearModelWeights = getWeightsData(linearModel, x_train, 'Linear_Model_Weight')
ridgeWeightsData = getWeightsData(RidgeRegression, x_train, 'Ridge_Weight')
lassoWeightsData = getWeightsData(LassoRegression, x_train, 'Lasso_weight')

finalWeights = pd.merge(linearModelWeights, ridgeWeightsData, on='Features')
finalWeights = pd.merge(finalWeights, lassoWeightsData, on='Features')



























