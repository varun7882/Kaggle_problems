# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 15:30:04 2018

@author: VaSrivastava
"""

import pandas as pd
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# SciKitLearn is a useful machine learning utilities library
import sklearn
# The sklearn dataset module helps generating datasets
from sklearn.preprocessing import OneHotEncoder
dataset=pd.read_csv("train.csv")
sub=pd.read_csv("test.csv")
ds=dataset.loc[:,['Pclass','Sex','Age','SibSp','Parch','Embarked']]
dssub=sub.loc[:,['Pclass','Sex','Age','SibSp','Parch','Embarked']]
ds.SibSp=ds.SibSp+ds.Parch+1
dssub.SibSp=dssub.SibSp+dssub.Parch+1
ds.Parch=0
dssub.Parch=0
ds.loc[ds['SibSp']==1,['Parch']]=1
dssub.loc[dssub['SibSp']==1,['Parch']]=1
pid=sub.PassengerId.values
#handling missing values
#for Age
ds.Age=ds.Age.fillna(ds.Age.median())
dssub.Age=dssub.Age.fillna(dssub.Age.median())
#for Embarked
ds.Embarked=ds.Embarked.fillna('S')
dssub.Embarked=dssub.Embarked.fillna('S')

ds.to_csv('preprocessedTrainData.csv',index=False)
dssub.to_csv('preprocessedTestData.csv',index=False)