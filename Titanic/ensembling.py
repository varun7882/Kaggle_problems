# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 18:46:32 2018

@author: VaSrivastava
"""

import pandas as pd
dsvm=pd.read_csv('result_svm.csv')
dxgb=pd.read_csv('result_xgb.csv')
dnn=pd.read_csv('result_8h.csv')
dnn.Survived+=dxgb.Survived+dsvm.Survived
dnn.loc[dnn['Survived']<2,'Survived']=0
dnn.loc[dnn['Survived']>=2,'Survived']=1
dnn.to_csv('resulten.csv',index=False)