# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:49:09 2020

@author: noama
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
df_train=pd.read_csv('C:/temp/exit/housing price/train (1).csv')
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(test[cols],size=1)
#histogram and normal probability plot
#sns.distplot(df_train['SalePrice']);
#fig = plt.figure()
print(df_train.size)
#drop outliners
df_train=df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)
#normalize the data
#df_train['SalePrice']=np.log1p(df_train['SalePrice'])
reg=LinearRegression().fit(df_train[['GrLivArea','LotArea']],df_train['SalePrice'])
reg_pred=reg.predict(df_train[['GrLivArea','LotArea']])
print(reg.score(df_train[['GrLivArea','LotArea']],df_train['SalePrice']))
print(mean_squared_error(reg_pred,df_train['SalePrice']))
error=reg_pred-df_train['SalePrice']
print(100*((df_train['SalePrice']-error)/df_train['SalePrice']).mean())
