# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 16:00:05 2018

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
from keras import regularizers
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


X_all=np.concatenate((X,X_sub),axis=0)
y=dataset.loc[:,'Survived'].values
#Handling categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_all[:, 1] = labelencoder_X.fit_transform(X_all[:, 1])
X_all[:, 5] = labelencoder_X.fit_transform(X_all[:, 5])
onehotencoder = OneHotEncoder(categorical_features=[0,5])
X_all = onehotencoder.fit_transform(X_all).toarray()
X=X_all[:891,:]
X_sub=X_all[891:,:]
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.33, random_state = 1)

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(26,input_dim=len(X_train[0]),activation='relu',init='uniform',kernel_regularizer=regularizers.l2(.5)))
model.add(Dense(1,activation='sigmoid',init='uniform'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=300,batch_size=50)

y_pred_val=model.predict(X_val)
y_pred_val[y_pred_val>=0.5]=1
y_pred_val[y_pred_val<0.5]=0
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, y_pred_val)
print ('confusion matrix :')
print (cm)
from sklearn.metrics import f1_score
print ('f-score(weighted) is : ')
print (f1_score(y_val,y_pred_val,average='weighted'))


y_sub=model.predict(X_sub)
y_sub[y_sub>=0.5]=1
y_sub[y_sub<0.5]=0
dct={'PassengerId':pid,'Survived':y_sub}
resdf=pd.DataFrame(data=dct)
resdf.to_csv('result_8hAdam.csv',index=False)