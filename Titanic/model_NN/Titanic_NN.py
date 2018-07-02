# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 18:28:21 2018

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

 
xh=[]

yf=[]
for h in range(2,25):
    xh.append(h)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(h), random_state=1)
    clf.fit(X_train, y_train)
    y_pred_train=clf.predict(X_train)
    cm = confusion_matrix(y_train, y_pred_train)
    #print 'confusion matrix :'
    #print cm
    #print 'f-scoreweighted) is : '
    f1train=f1_score(y_train,y_pred_train,average='weighted')
    #print f1train
    y_pred_val=clf.predict(X_val)
    cm = confusion_matrix(y_val, y_pred_val)
    #print 'confusion matrix :'
    #print cm
    print ('f-scoreweighted) is : ')
    f1val=f1_score(y_val,y_pred_val,average='weighted')
    #print f1val
    print (f1val)
    print ('hidden ',h,'^')
    yf.append(f1val)
plt.xlabel('Hidden layer neurons')
plt.ylabel('fscore')
plt.title('hidden layer neurons vs fscore')
plt.plot(xh,yf)
plt.savefig('hidden layer neurons vs fscore(train_validation).png')
plt.show()

 
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(19), random_state=1)
clf.fit(X_train, y_train)
y_pred=clf.predict(sub.values)