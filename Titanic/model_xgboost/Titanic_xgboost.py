# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:46:25 2018

@author: VaSrivastava
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ataset=pd.read_csv("../preprocessedTrain.csv")
sub=pd.read_csv("../preprocessedTest.csv")
X=dataset.loc[:,['Pclass','Sex','Age','Fare','Embarked','IsAlone','Title']].values
y=dataset.loc[:,['Survived']].values
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.33, random_state = 1)
clf = XGBClassifier(learning_rate =1,
 n_estimators=10000,
 max_depth=500,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
clf.fit(X_train, y_train)

y_pred_val=clf.predict(X_val)
print ('f-scoreweighted) is : ')
f1val=f1_score(y_val,y_pred_val,average='weighted')
print (f1val)
y_pred=clf.predict(X_test)
pid=sub.loc[:,['PassengerId']].values.reshape(-1)
dct={'PassengerId':pid,'Survived':y_pred}
resdf=pd.DataFrame(data=dct)
resdf.to_csv('result_xgb.csv',index=False)