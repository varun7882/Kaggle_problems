# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 22:10:12 2018

@author: VaSrivastava
"""

import pandas as pd
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab as py
# SciKitLearn is a useful machine learning utilities library
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
# The sklearn dataset module helps generating datasets

dataset=pd.read_csv("../preprocessedTrain.csv")
sub=pd.read_csv("../preprocessedTest.csv")
X=dataset.loc[:,['Pclass','Sex','Age','Fare','Embarked','IsAlone','Title']].values
y=dataset.loc[:,['Survived']].values
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.33, random_state = 1)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(17), random_state=1)
clf.fit(X, y)
X_test=sub.loc[:,['Pclass','Sex','Age','Fare','Embarked','IsAlone','Title']].values
y_pred=clf.predict(X_test)
pid=sub.loc[:,['PassengerId']].values.reshape(-1)
dct={'PassengerId':pid,'Survived':y_pred}
resdf=pd.DataFrame(data=dct)
resdf.to_csv('result_8hnew.csv',index=False)