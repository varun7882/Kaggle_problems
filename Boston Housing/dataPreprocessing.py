# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 23:04:02 2018

@author: VaSrivastava
"""

import numpy as np
import pandas as pd


data=pd.read_csv('train.csv')
target=data.medv.values
data_x=data.drop(['ID','medv'],axis=1)
#Normalizing x
data_x=(data_x-data_x.mean())/data_x.std()
data_y=pd.DataFrame(target,columns=['medv'])
data_x.to_csv('normalizedX.csv',index=False)
data_y.to_csv('targetY.csv',index=False)

test_data=pd.read_csv('test.csv')
X_test=test_data.drop(['ID'],axis=1)
X_test=(X_test-X_test.mean())/X_test.std()
X_test.to_csv('normalized_testX.csv',index=False)

