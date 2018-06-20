# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 23:40:59 2018

@author: VaSrivastava
"""

import numpy as np
import pandas as pd
import sklearn as skl
import tensorflow as tf

data_X=pd.read_csv('normalizedX.csv')
data_X.insert(0,'bias',1)
test_data=pd.read_csv('normalized_testX.csv')
test_data.insert(0,'bias',1)
X_test=test_data.values
data_y=pd.read_csv('targetY.csv')
X=data_X.values
y=data_y.values

xT=np.transpose(X)
tmp_x=np.matmul(xT,X)
invX=np.linalg.inv(tmp_x)
tmp2_x=np.matmul(invX,xT)
theta=np.matmul(tmp2_x,y)

y_test=np.matmul(X_test,theta)

IDy=pd.read_csv('test.csv')
dictd={'ID':IDy.loc[:,'ID'].values,'medv':y_test.reshape(-1)}
ans=pd.DataFrame(data=dictd)
ans.to_csv('NormalizedEquationAttempt1.csv',index=False)