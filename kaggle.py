# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 00:27:54 2020

@author: noama
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


#read df:
data=pd.read_csv('c:/temp/exit/train.csv')
#extracting data for test
result=data['Survived']
data=data.drop('Survived',1)
#making embarked into 3 binary categories:
ports=pd.get_dummies(data.Embarked,prefix='Embarked')
#adding the new binary categories
data=data.join(ports)
data.drop(['Embarked'],axis=1,inplace=True)
#making the sex 1 for woman 0 for man
data.Sex=data.Sex.map({'male':0,'female':1})
X=data
#dropping unnccesery data for this model
X.drop(['Cabin'], axis=1, inplace=True) 

X.drop(['Ticket'], axis=1, inplace=True) 
X.drop(['Name'], axis=1, inplace=True) 
X.drop(['PassengerId'], axis=1, inplace=True)
X.Age.fillna(X.Age.mean(),inplace=True)
#X_train,X_valid,Y_train,Y_valid=train_test_split(X,result,test_size=0.2,random_state=7)
model=LogisticRegression(penalty='l1', solver='liblinear')
model.fit(X,result)

print(model.score(X,result))
print(X.info())
print((model.coef_)) 
##read df:
#data=pd.read_csv('c:/temp/exit/test.csv')
##extracting data for test
##making embarked into 3 binary categories:
#ports=pd.get_dummies(data.Embarked,prefix='Embarked')
##adding the new binary categories
#data=data.join(ports)
#data.drop(['Embarked'],axis=1,inplace=True)
##making the sex 1 for woman 0 for man
#data.Sex=data.Sex.map({'male':0,'female':1})
#X=data
##dropping unnccesery data for this model
#X.drop(['Cabin'], axis=1, inplace=True) 
#X.drop(['Ticket'], axis=1, inplace=True) 
#z=X['PassengerId']
#X.drop(['Name'], axis=1, inplace=True) 
#X.drop(['PassengerId'], axis=1, inplace=True)
#X.Age.fillna(X.Age.mean(),inplace=True)
#X.Fare.fillna(X.Fare.mean(),inplace=True)
#f=pd.DataFrame(model.predict(X))
#z=f.join(z)
#
#z.to_csv('c:/temp/exit/noam.csv')
#
#o=pd.read_csv('c:/temp/exit/noam.csv')
#o.drop('Unnamed: 0',inplace=True,axis=1)
#o.to_csv('c:/temp/exit/noam.csv')
#
